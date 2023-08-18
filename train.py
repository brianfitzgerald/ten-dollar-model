import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import CSVLogger
import torchvision.transforms as transforms
import fire
from sentence_transformers import SentenceTransformer, util
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor

torch.manual_seed(0)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.upsample_conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.upsample_conv1(x)
        out = self.bn1(out)
        out = self.upsample_conv2(out)
        out += residual
        out = self.relu(out)
        return out


class PixelDataset(Dataset):
    def __init__(self, data_root, num_colors: int = 256, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_list = [
            filename for filename in os.listdir(data_root) if filename.endswith(".png")
        ]
        self.num_colors: int = num_colors

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_filename = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_filename)

        image = Image.open(image_path)
        image_tensor = ToTensor()(image).unsqueeze(0)

        one_hot_encoded = F.one_hot(
            image_tensor.to(torch.int64), num_classes=self.num_colors
        )

        image_name = os.path.splitext(image_filename)[0]

        if self.transform:
            one_hot_encoded = self.transform(one_hot_encoded)

        return one_hot_encoded, image_name


class Generator(nn.Module):
    def __init__(
        self,
        noise_emb_size: int = 5,
        out_size: int = 16,
        num_colors: int = 256,
        num_residual_blocks: int = 6,
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.text_emb_size: int = 384
        self.num_colors = num_colors

        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.reshape_layer = nn.Linear(
            noise_emb_size + self.text_emb_size, 2 * 2 * num_colors
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(3, 3, 7) for x in range(num_residual_blocks)]
        )
        self.out_conv = nn.Conv2d(out_size, out_size, out_size)

    def forward(self, image, caption_enc):
        batch_size = image.shape[0]
        noise = torch.randn((batch_size, self.noise_emb_size))
        input_emb = torch.cat([noise, caption_enc], 1).to("cuda")
        x = torch.relu(self.reshape_layer(input_emb))
        breakpoint()
        x = x.view(batch_size, 2, 2, self.num_colors)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.residual_blocks(x)
        x = self.out_conv(x)
        return x


class GeneratorModule(pl.LightningModule):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ).eval()
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

    def training_step(self, batch):
        image, caption = batch
        caption_enc = torch.from_numpy(self.sentence_encoder.encode(caption))
        preds = self.generator(image, caption_enc)
        loss = self.loss_fn(preds, image)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        users, items, ratings = batch
        preds = self.generator(users, items)
        loss = self.loss_fn(preds.squeeze(1), ratings)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(use_wandb: bool = False):
    dataset = PixelDataset("./spritesheets/food")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=1)
    model = GeneratorModule(Generator())
    logger = None
    if use_wandb:
        logger = WandbLogger(project="recsys")
        logger.watch(model)
    else:
        logger = None
    trainer = pl.Trainer(logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    fire.Fire(main)
