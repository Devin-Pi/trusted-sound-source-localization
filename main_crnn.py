# Imports for plotting
import os
from copy import deepcopy

import lightning as l
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from matplotlib.colors import to_rgb
from packaging.version import Version
from torch.functional import F
from torch.utils.data import DataLoader

from dataloader.dataset_cls import TSSLDataSet
from model.module import PredDOA
from model.tcrnn import CRNN

plt.set_cmap('cividis')

matplotlib.rcParams['lines.linewidth'] = 2.0

sns.reset_orig()


class DataModule(l.LightningDataModule):
    def __init__(self, data_dir: str = "/TSSL/data/", batch_size: tuple = [2, 1], num_workers: int = 8):
        super().__init__()
        """this class is for the datamodule
        Args:
            data_dir (str): the path of the data
            batch_size (list): a list of batch size for train and test [2, 2]
            num_workers (int, optional): the value of the num_workers. Defaults to 0.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str):
        print(stage)
        if stage == "fit":
            self.dataset_train = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "train"),
                num_data=5000,
            )
            self.dataset_val = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "dev"),
                num_data=998,
            )
        elif stage == "test":
            self.dataset_test = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "test"),
                num_data=5000
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size[0],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size[1],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size[1],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )


class TrustedRCNN(l.LightningModule):
    def __init__(
        self,
        input_dim: int = 4,
        num_classes: int = 180,
        hidden_dim: int = 64,
        lr=0.0005,
        tar_useVAD: bool = True,
        ch_mode: str = 'MM',
        fs: int = 16000,
        method_mode: str = 'IDL',
        source_num_mode: str = 'KNum',
        max_num_sources: int = 1,
        return_metric: bool = True,
        compile: bool = False,
        device: str = "cuda",
        lamdba_peochs: int = 10,
    ):
        super().__init__()
        # Model init
        self.model = CRNN(
            max_num_sources=max_num_sources,
            input_dim=input_dim,
            cnn_dim=hidden_dim,
            num_classes=num_classes,
        )

        torch.set_float32_matmul_precision('medium')

        if compile:
            print("Compiling the model!")
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.model = torch.compile(self.model)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['model'])
        self.tar_useVAD = tar_useVAD
        self.method_mode = method_mode
        self.dev = device
        self.source_num_mode = source_num_mode
        self.max_num_sources = max_num_sources
        self.ch_mode = ch_mode
        self.lamdba_epochs = lamdba_peochs

        self.fre_max = fs / 2
        self.return_metric = return_metric
        self.get_metric = PredDOA()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8988, last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                # 'monitor': 'valid/loss',
            }
        }

    def training_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]  # [2, 4, 256, 299] bs, c, f, t
        gt_batch = batch[1]
        pred_batch = self.model(mic_sig_batch)

        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        self.log("train/loss", loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch)
        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        # loss = self.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("valid/loss", loss, sync_dist=True, on_epoch=True)

        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        for m in metric:
            self.log('valid/'+m, metric[m].item(),
                     sync_dist=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch)  # [2, 24, 512]
        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)
        # loss = self.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        # print(metric)
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)

    def predict_step(self, batch, batch_idx: int):

        mic_sig_batch = batch[0]
        pred_batch = self.forward(mic_sig_batch)

        return pred_batch

# loss function
    def KL(self, alpha, c):
        beta = torch.ones((1, c)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0),
                       dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def ce_kl_loss(self, p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)  # [48, 180]
        A = torch.sum(label * (torch.digamma(S) -
                      torch.digamma(alpha)), dim=1, keepdim=True)
        # annealing_coef = min(1, global_step / annealing_step)
        alp = E * (1 - label) + 1
        annealing_coef = torch.tensor(
            0.01, dtype=torch.float32).to(self.device)
        B = annealing_coef * self.KL(alp, c)
        # return (A + B)
        return A

    def ce_loss_uncertainty(self, pred_batch=None, gt_batch=None, current_epoch=None):
        """ 
                Function: ce loss for uncertainty
                Args:
                        pred_batch: doa
                        gt_batch: dict{'doa'}
                Returns:
                        loss
        """
        # self.log("NO.CURRENT_EPOCH",current_epoch, sync_dist=True,on_epoch=True)
        nb, nt, _ = pred_batch.shape
        pred_batch = pred_batch.reshape(nb*nt, -1)  # [48, 180]

        gt_doa = gt_batch['doa'] * 180 / np.pi
        gt_doa = gt_doa[:, :, 1, :].type(
            torch.LongTensor).to(self.device)  # [2, 24, 1]
        gt_doa = gt_doa.reshape(nb*nt)  # [48]

        # obtain evidence
        # evidence = F.relu(pred_batch) # obtain evidence
        evidence = torch.exp(torch.clamp(pred_batch, -10, 10))

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = 180 / S
        edl_loss = self.ce_kl_loss(
            gt_doa,
            alpha,
            180,
            current_epoch,
            self.lamdba_epochs,
        )
        loss_all = torch.mean(edl_loss)
        return loss_all, evidence, U

    def ce_loss(self, pred_batch=None, gt_batch=None):
        """ 
                Function: ce loss
                Args:
                        pred_batch: doa
                        gt_batch: dict{'doa'}
                Returns:
                        loss
        """
        pred_doa = pred_batch
        gt_doa = gt_batch['doa'] * 180 / np.pi
        gt_doa = gt_doa[:, :, 1, :].type(torch.LongTensor).to(self.device)
        nb, nt, _ = pred_doa.shape
        pred_doa = pred_doa
        loss = torch.nn.functional.cross_entropy(
            pred_doa.reshape(nb*nt, -1), gt_doa.reshape(nb*nt))
        return loss


def cli_main():

    cli = LightningCLI(
        TrustedRCNN,
        DataModule,
        seed_everything_default=1744,
        save_config_kwargs={'overwrite': True},
        # parser_kwargs={"default_config_files": ["config/default.yaml"],
        #    "parser_mode": "omegaconf"
        #    },
    )
    # model = TransformerPredictor()
    # data = DataModule()

    # trainer = l.Trainer(
    #     accelerator="auto",
    #     fast_dev_run=10,
    # )
    # trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
