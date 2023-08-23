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
from typing import List, Tuple
import shutil
import imageio

torch.manual_seed(0)


def encode_image(image: Image.Image, palette: np.ndarray) -> torch.Tensor:
    def find_closest_color(pixel):
        distances = np.sum((palette - pixel) ** 2, axis=1)
        return np.argmin(distances)

    quantized_image = np.apply_along_axis(find_closest_color, 2, image)
    one_hot_encoded = F.one_hot(torch.tensor(quantized_image), len(palette))
    return one_hot_encoded


def decode_image_batch(
    image_tensor_batch: torch.Tensor, palette: np.ndarray
) -> List[Image.Image]:
    out_imgs = []
    for batch_idx in range(image_tensor_batch.shape[0]):
        img_tensor = image_tensor_batch[batch_idx].argmax(dim=-1).cpu()
        rgb_image = palette[img_tensor]
        pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
        out_imgs.append(pil_image)

    return out_imgs


def image_grid(image_list: List[List[Image.Image]]) -> Image.Image:
    num_rows, num_cols = len(image_list), len(image_list[0])
    image_width, image_height = image_list[0][0].size

    grid_width = num_cols * image_width
    grid_height = num_rows * image_height

    grid_image = Image.new("RGB", (grid_width, grid_height))

    for row in range(num_rows):
        for col in range(num_cols):
            x_offset = col * image_width
            y_offset = row * image_height
            grid_image.paste(image_list[row][col], (x_offset, y_offset))
    return grid_image


class PixelDataset(Dataset):
    def __init__(self, data_root, palette):
        self.data_root = data_root
        self.image_list = [
            filename for filename in os.listdir(data_root) if filename.endswith(".png")
        ]
        self.palette_img = palette
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_filename = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_filename)

        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        image_tensor = encode_image(image, self.palette_img)

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
        num_filters: int = 512,
        num_residual_blocks: int = 8,
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
            self.results_table = wandb.Table(columns=["results"])
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

    def eval_step(self, batch, epoch: int):
        image, caption = batch
        image = image.to(self.device)
        caption_enc = torch.from_numpy(self.sentence_encoder.encode(caption)).to(
            self.device
        )
        preds = self.generator(image, caption_enc)
        input_images = decode_image_batch(image, self.palette)
        preds_reordered = preds.permute(0, 2, 3, 1)
        pred_images = decode_image_batch(preds_reordered, self.palette)
        results_grid = image_grid([input_images, pred_images])
        if self.use_wandb:
            self.results_table.add_data([wandb.Image(results_grid)])
            wandb.log({"results": self.results_table})
        results_grid.save(os.path.join("debug_images", f"results_epoch_{epoch}.png"))


def main(use_wandb: bool = False, num_epochs: int = 5000, eval_every: int = 10):
    if use_wandb:
        wandb.init(project="ten-dollar-model")

    shutil.rmtree("debug_images", ignore_errors=True)
    Path("debug_images").mkdir(exist_ok=True)

    color_palette = np.array(pico_rgb_palette) / 255.0
    num_colors = len(color_palette)

    # Test encoding / decoding
    enc_test_image = imageio.imread('./spritesheets/food/Wine.png', pilmode="RGB").astype(float) / 255.0
    encoded = encode_image(enc_test_image, color_palette)
    decoded = decode_image_batch(encoded.unsqueeze(0), color_palette)
    decoded[0].save(os.path.join("debug_images", "decoded.png"))

    dataset = PixelDataset("./spritesheets/food", color_palette)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    # test encoding / decoding
    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=4)
    device = torch.device("cuda")
    model = GeneratorModule(
        device, Generator(device, num_colors=num_colors), color_palette, use_wandb
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for i in range(num_epochs):
        for j, batch in enumerate(train_dataloader):
            loss = model.training_step(batch)
            loss.backward()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm()

            print("Total norm: ", total_norm)
            print(f"Epoch {i}, batch {j}, loss {loss.item()}, total norm: {total_norm}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
        if i % eval_every == 0:
            for j, batch in enumerate(test_dataloader):
                print("Running eval..")
                model.eval_step(batch, i)
                break


if __name__ == "__main__":
    fire.Fire(main)
