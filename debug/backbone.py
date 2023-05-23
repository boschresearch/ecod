# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl


from ecod.models.backbones.resnet import ResNetBackbone, ResNetBackboneSmall
from ecod.models.necks.conv import ConvNeck
from ecod.data.mnist.dataset import MnistDataset
from ecod.utils.models import get_output_shape

torch.backends.cudnn.benchmark = True

epochs = 10
device = "cuda"
n_classes = 10
#shape = (160, 160)
shape = (360, 360)
batch_size = 64
num_workers = 8
random_resize = True
pad = True


class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = ResNetBackbone("resnet18", 1, pool=True, pretrained=False)
        self.backbone = ResNetBackboneSmall(1)
        self.out_shape = get_output_shape(self.backbone, (1, *shape))
        print("backbone output shape: ", self.out_shape)
        self.neck = ConvNeck(self.out_shape[0], [128, 128])
        self.neck_out_shape = self.neck.get_output_shape(self.out_shape)
        print("neck output shape: ", self.neck_out_shape)
        self.head = torch.nn.Linear(int(np.prod(self.neck_out_shape[-1][:1])), n_classes)

    def forward(self, x):
        x = self.backbone(x)
        outs = self.neck(x)
        x = outs[-1]
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


class LitClf(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # my_script_module = torch.jit.script(Clf())
        # self.net = my_script_module
        self.net = Clf()
        self.loss = torch.nn.NLLLoss()
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        out = self.logsoft(model(imgs))
        batch_loss = self.loss(out, labels)
        self.log("train_loss", batch_loss)
        return batch_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        return optimizer

    def valtest_step(self, batch, batch_idx):
        imgs, labels = batch
        y_hat = torch.argmax(self(imgs), 1)
        corrects = (y_hat == labels).to(torch.float32)
        return {"correct": corrects}

    def validation_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx)

    def valtest_epoch_end(self, outputs, valtest):
        acc = torch.cat([out["correct"] for out in outputs])
        acc = acc.mean().cpu().numpy().item()
        self.log(f"{valtest}_acc", acc)
        print("Accuracy", acc)

    def validation_epoch_end(self, outputs):
        return self.valtest_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.valtest_epoch_end(outputs, "test")

    def train_dataloader(self):
        dset = MnistDataset(shape, "train", pad=pad, random_resize=random_resize)
        dloader_train = DataLoader(
            dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True
        )
        return dloader_train

    def val_dataloader(self):
        dloader_test = DataLoader(
            MnistDataset(shape, "test", pad=pad, random_resize=random_resize),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dloader_test

    def test_dataloader(self):
        dloader_test = DataLoader(
            MnistDataset(shape, "test", pad=pad, random_resize=random_resize),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dloader_test


model = LitClf()
pl.utilities.model_summary.ModelSummary(model, max_depth=10)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, check_val_every_n_epoch=1)
trainer.fit(model=model)
