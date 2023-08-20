import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import fire
from sentence_transformers import SentenceTransformer, util
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from PIL import Image, ImagePalette
from torchvision.transforms import ToPILImage
import numpy as np
import wandb
from pathlib import Path
from color_bank import pico_rgb_palette
from typing import List

torch.manual_seed(0)


def encode_image(image: Image.Image, palette: Image.Image):
    image_p = image.quantize(palette=palette)
    quantized_tensor = torch.from_numpy(np.array(image_p)).long()
    one_hot_encoded = F.one_hot(quantized_tensor, 256)
    return one_hot_encoded


def decode_image_batch(image_tensor: torch.Tensor, palette: Image.Image) -> List[Image.Image]:
    out_imgs = []
    for batch_idx in range(image_tensor.shape[0]):
        image_tensor_sample = image_tensor[batch_idx]
        image_tensor_sample = image_tensor_sample.argmax(-1)
        quantized = Image.new(
            "P", (image_tensor_sample.size(0), image_tensor_sample.size(0))
        )
        quantized.putdata(image_tensor_sample.flatten().tolist())
        quantized.putpalette(palette.palette)
        quantized = quantized.convert("RGB")
        out_imgs.append(quantized)
    return out_imgs


class PixelDataset(Dataset):
    def __init__(
        self,
        data_root,
        palette: Image.Image,
        num_colors: int = 16,
    ):
        self.data_root = data_root
        self.image_list = [
            filename for filename in os.listdir(data_root) if filename.endswith(".png")
        ]
        self.num_colors: int = num_colors
        self.palette = palette

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_filename = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = encode_image(image, self.palette)

        image_name = os.path.splitext(image_filename)[0]

        return image_tensor, image_name


class ResidualBlock(nn.Module):
    def __init__(
        self, num_filters: int = 128, kernel_size: int = 7, upsampling=False
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.upsampling = upsampling
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.upsampling:
            x = self.upsample(x)
        residual = x
        x = self.layers(x)
        return residual + x


class Generator(nn.Module):
    def __init__(
        self,
        device: torch.device,
        noise_emb_size: int = 5,
        num_colors: int = 16,
        num_filters: int = 128,
        num_residual_blocks: int = 6,
        kernel_size: int = 7,
        conv_size: int = 4,
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.text_emb_size: int = 384
        self.num_filters = num_filters
        self.device = device
        self.num_colors = num_colors
        self.conv_size = conv_size

        self.reshape_layer = nn.Linear(
            noise_emb_size + self.text_emb_size,
            self.conv_size * self.conv_size * num_filters,
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(self.num_filters, kernel_size, i < 2)
                for i in range(num_residual_blocks)
            ]
        )
        self.pad = nn.ZeroPad2d(1)
        self.out_conv = nn.Conv2d(
            in_channels=self.num_filters, out_channels=self.num_colors, kernel_size=3
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image, caption_enc):
        batch_size = image.shape[0]
        noise = torch.randn((batch_size, self.noise_emb_size)).to(self.device)
        input_emb = torch.cat([noise, caption_enc], 1).to(self.device)
        x = self.reshape_layer(input_emb)
        x = x.view(-1, self.num_filters, self.conv_size, self.conv_size)
        x = self.residual_blocks(x)
        x = self.pad(x)
        x = self.out_conv(x)
        return self.softmax(x)


class GeneratorModule(nn.Module):
    def __init__(
        self,
        device: torch.device,
        generator: Generator,
        palette: Image.Image,
        use_wandb: bool = False,
    ):
        super().__init__()
        self.device = device
        self.generator = generator.to(device)
        self.loss_fn = torch.nn.NLLLoss()
        self.sentence_encoder = (
            SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            .to(device)
            .eval()
        )
        self.use_wandb = use_wandb
        self.results_table = None
        self.palette = palette
        if self.use_wandb:
            wandb.watch(self.generator)
            self.results_table = wandb.Table(columns=["image", "sample"])
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

    def training_step(self, batch):
        image, caption = batch
        image = image.to(self.device)
        caption_enc = torch.from_numpy(self.sentence_encoder.encode(caption)).to(
            self.device
        )
        preds = self.generator(image, caption_enc)
        image_classes = image.argmax(dim=-1)
        loss = self.loss_fn(torch.log(preds), image_classes)
        if self.use_wandb:
            wandb.log({"loss": loss.item()})
        return loss

    def eval_step(self, batch):
        image, caption = batch
        image = image.to(self.device)
        caption_enc = torch.from_numpy(self.sentence_encoder.encode(caption)).to(
            self.device
        )
        preds = self.generator(image, caption_enc)
        input_images = decode_image_batch(image, self.palette)
        preds_reordered = preds.permute(0,2,3,1)
        pred_images = decode_image_batch(preds_reordered, self.palette)
        if self.use_wandb:
            input_images = [wandb.Image(im) for im in input_images]
            pred_images = [wandb.Image(im) for im in pred_images]
            self.results_table.add_data(input_images, pred_images)
            wandb.log({"results": self.results_table})
        else:
            Path("debug_images").mkdir(exist_ok=True)
            for i in range(image.shape[0]):
                input_images[i].save(os.path.join("debug_images", f"input_{i}.png"))
                pred_images[i].save(os.path.join("debug_images", f"preds_{i}.png"))


def main(use_wandb: bool = False, num_epochs: int = 5000, log_every: int = 1):
    num_colors = 256
    if use_wandb:
        wandb.init(project="ten-dollar-model")

    image = Image.open("./spritesheets/food/Apple Pie.png").convert("RGB")

    red_values = [r for r, _, _ in pico_rgb_palette]
    green_values = [g for _, g, _ in pico_rgb_palette]
    blue_values = [b for _, _, b in pico_rgb_palette]

    # Flattened palette with R values followed by G values and then B values
    flattened_palette = red_values + green_values + blue_values

    pico_palette = ImagePalette.ImagePalette(
        palette=flattened_palette, size=len(flattened_palette)
    )
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(pico_palette)

    # Test encoding / decoding
    # encoded = encode_image(image, palette_img)
    # decoded = decode_image_batch(encoded.unsqueeze(0), palette_img.palette)
    # decoded[0].save(os.path.join("debug_images", "decoded.png"))

    dataset = PixelDataset("./spritesheets/food", palette_img, num_colors)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    # test encoding / decoding
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=1)
    device = torch.device("cuda")
    model = GeneratorModule(
        device, Generator(device, num_colors=num_colors), palette_img, use_wandb
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(num_epochs):
        for j, batch in enumerate(train_dataloader):
            loss = model.training_step(batch)
            print(i, j, loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if i % log_every == 0:
            for j, batch in enumerate(test_dataloader):
                model.eval_step(batch)
                break


if __name__ == "__main__":
    fire.Fire(main)
