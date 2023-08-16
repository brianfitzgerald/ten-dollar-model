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

from dataset import MovieLens20MDataset


torch.manual_seed(0)


class Generator(nn.Module):
    def __init__(
        self, noise_emb_size: int = 5, out_size: int = 16, upsample_size: int = 256
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.reshape_layer = nn.Linear(noise_emb_size + 384, 2 * 2 * 256)
        self.upsample_conv1 = nn.Conv2d(2, 2, upsample_size, upsample_size)
        self.upsample_conv2 = nn.Conv2d(4, 4, upsample_size, upsample_size)

    def forward(self, image, caption_enc):
        bsz = image.shape[0]
        noise = torch.randn((bsz, self.noise_emb_size))
        input_emb = torch.cat([noise, caption_enc])
        x = self.reshape_layer(input_emb)


class GeneratorModule(pl.LightningModule):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )

    def training_step(self, batch):
        image, caption = batch
        caption_enc = self.sentence_encoder.encode(caption)
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
    dataset = MovieLens20MDataset("ml-25m/ratings.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    no_users, no_movies = dataset.no_movies, dataset.no_users
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=30)
    val_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=30)
    model = GeneratorModule(Generator(no_movies, no_users))
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
