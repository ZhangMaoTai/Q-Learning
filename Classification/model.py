import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from utils.const import *


class ClassificationModel(nn.Module):
    def __init__(self,
                 input_dim: int = MAX_WORD_LEN,
                 output_dim: int = OUTPUT_DIM,
                 embedding_dim: int = EMBEDDING_DIM
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(28, self.embedding_dim)
        self._init_weights(self.embedding)

        self.letter_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 32),
            nn.SELU(),
            nn.Dropout(0.3),
        )
        self._init_weights(self.letter_fc)

        self.combine_fc = nn.Sequential(
            nn.Linear(self.input_dim * 32, 1024),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(128, self.output_dim)
        )
        self._init_weights(self.combine_fc)

    def forward(self, state):
        # state: batch, MAX_WORD_LEN
        state = self.embedding(state)               # batch, MAX_WORD_LEN, embedding_dim

        state = self.letter_fc(state)               # batch, MAX_WORD_LEN, 32
        state = state.view(state.size(0), -1)       # batch, MAX_WORD_LEN * 32

        return self.combine_fc(state)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.001)

        elif isinstance(module, nn.Embedding):
            padding_idx = 1
            module.weight.data.normal_(mean=0.0, std=0.1)
            module.weight.data[padding_idx].zero_()


def total_loss(y_hat, y_true):
    # y_true: batch，26
    return nn.BCEWithLogitsLoss()(y_hat, y_true)


def calculate_f1(y_pred , y_true):
    return f1_score(y_true.view(-1), y_pred.view(-1))

