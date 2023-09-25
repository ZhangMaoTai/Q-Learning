import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import get_scheduler

from tqdm.auto import tqdm

import wandb
import os

from utils.util import wandb_start
from Classification.model import total_loss, calculate_f1


class Trainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 eval_dataloader,

                 epochs,
                 learning_rate,
                 num_warmup_steps
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate)
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.epochs * len(self.train_dataloader)
        )

        # init wandb
        self.writer = wandb_start(True)

    def train(self,
              print_freq,
              save_dir,
              max_norm
              ):
        torch.cuda.empty_cache()
        progress_bar = tqdm(range(self.epochs * len(self.train_dataloader)))
        f1_list = []

        self.model.zero_grad()
        self.optimizer.zero_grad()

        for epoch in range(self.epochs):
            self.model.train()

            for updates, (x_train, y_train) in enumerate(self.train_dataloader):
                x_train = {k: v.to(self.device) for k, v in x_train.items()}
                y_train = {k: v.to(self.device) for k, v in y_train.items()}

                y_hat = self.model(x_train)
                loss = total_loss(y_hat, y_train)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

                self.writer.add_scalar("losses/Train_LOSS", loss.item(), updates)

                if updates % print_freq == 0 and updates != 0:
                    test_loss, F1 = self.eval()

                    self.writer.add_scalar("losses/Test_LOSS", test_loss, updates)
                    self.writer.add_scalar("losses/F1", F1, updates)

                    if F1 > max(f1_list):
                        torch.save(self.model.state_dict(), os.path.join(save_dir, "best_{}.pth".format(updates)))
                    torch.save(self.model.state_dict(), os.path.join(save_dir, "model_{}.pth".format(updates)))

                    f1_list.append(F1)
                    self.model.train()

        self.writer.close()
        wandb.finish()

    def eval(self):
        self.model.eval()

        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(self.eval_dataloader):
                x_test = {k: v.to(self.device) for k, v in x_test.items()}
                output = self.model(x_test).to('cpu')

                y_test = {k: v.to('cpu') for k, v in y_test.items()}

                if i == 0:
                    y_hat_total = output
                    y_true_total = y_test
                else:
                    y_hat_total = torch.cat((y_hat_total, output), axis=0)
                    y_true_total = torch.cat((y_true_total, y_test), axis=0)

        loss_test = nn.BCEWithLogitsLoss()(y_hat_total, y_true_total).item()

        y_hat_total = nn.Sigmoid()(y_hat_total)
        y_test_pred = torch.where(y_hat_total > 0.5, 1., 0.)
        F1 = calculate_f1(y_test_pred, y_true_total)

        return loss_test, F1
