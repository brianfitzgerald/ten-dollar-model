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
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from color_bank import color_bank_hex, hex_to_rgb

torch.manual_seed(0)

color_bank = torch.tensor([hex_to_rgb(x) for x in color_bank_hex])

class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int = 1
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


def encode_image(image: Image.Image, num_colors: int):
    image_tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32)  # Convert to tensor
    height, width = image.size

    expanded_colors = color_bank.view(num_colors, 1, 1, 3)
    expanded_image = image_tensor.view(1, height, width, 3)
    distances = torch.norm(expanded_image - expanded_colors, dim=3)  # Euclidean distances

    nearest_color_indices = torch.argmin(distances, dim=0).float()

    return nearest_color_indices


class PixelDataset(Dataset):
    def __init__(self, data_root, num_colors: int = 256):
        self.data_root = data_root
        self.image_list = [
            filename for filename in os.listdir(data_root) if filename.endswith(".png")
        ]
        self.num_colors: int = num_colors

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_filename = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = encode_image(image, self.num_colors)

        image_name = os.path.splitext(image_filename)[0]

        return image_tensor, image_name


class Generator(nn.Module):
    def __init__(
        self,
        device: torch.device,
        noise_emb_size: int = 5,
        out_size: int = 16,
        num_colors: int = 256,
        num_residual_blocks: int = 6,
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.text_emb_size: int = 384
        self.num_colors = num_colors
        self.device = device
        self.out_size = out_size

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.reshape_layer = nn.Linear(
            noise_emb_size + self.text_emb_size, 2 * 2 * num_colors
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.num_colors, self.num_colors) for _ in range(num_residual_blocks)]
        )
        self.out_conv = nn.ConvTranspose2d(self.num_colors, self.num_colors, kernel_size=4, stride=4, padding=0)

    def forward(self, image, caption_enc):
        batch_size = image.shape[0]
        noise = torch.randn((batch_size, self.noise_emb_size)).to(self.device)
        input_emb = torch.cat([noise, caption_enc], 1).to(self.device)
        x = torch.relu(self.reshape_layer(input_emb))
        x = x.view(batch_size, self.num_colors, 2, 2)
        x = self.upsample1(x)
        x = self.residual_blocks(x)
        x = self.out_conv(x)
        return x


class GeneratorModule(nn.Module):
    def __init__(self, device: torch.device, generator: Generator):
        super().__init__()
        self.device = device
        self.generator = generator.to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sentence_encoder = (
            SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            .to(device)
            .eval()
        )
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

    def training_step(self, batch):
        image, caption = batch
        image = image.to(self.device)
        caption_enc = torch.from_numpy(self.sentence_encoder.encode(caption)).to(
            self.device
        )
        preds = self.generator(image, caption_enc)
        preds_reshaped = preds.permute(0,2,3,1).reshape(-1, self.generator.num_colors).to(self.device)
        image_reshaped = image.view(-1).type(torch.LongTensor).to(self.device)
        loss = self.loss_fn(preds_reshaped, image_reshaped)
        return loss

def main(use_wandb: bool = False, num_epochs: int = 5):
    dataset = PixelDataset("./spritesheets/food")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=1)
    device = torch.device("cuda")
    model = GeneratorModule(device, Generator(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(num_epochs):
        for j, batch in enumerate(train_dataloader):
            loss = model.training_step(batch)
            loss.backward()
            print(loss)
            optimizer.step()
            


if __name__ == "__main__":
    fire.Fire(main)
