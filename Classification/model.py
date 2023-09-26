import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from utils.const import *


class BiGRU2(nn.Module):
    def __init__(self,
                 input_dim: int = MAX_WORD_LEN,
                 output_dim: int = OUTPUT_DIM,
                 embedding_dim: int = EMBEDDING_DIM):
        super(BiGRU2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_sizes = [128, 256, 512, 1024, 1024, 1024]

        self.embedding = nn.Embedding(28, self.embedding_dim)
        self._init_weights(self.embedding)

        self.grus = nn.ModuleList()
        for i in range(len(self.hidden_sizes)):
            input_dim = self.embedding_dim if i == 0 else self.hidden_sizes[i - 1] * 2
            self.grus.append(
                nn.GRU(input_dim, self.hidden_sizes[i],
                       num_layers=1, bidirectional=True, batch_first=True, bias=True)
            )
        for m in self.grus:
            self._init_weights(m)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1]*2, 1024),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 26)
        )
        self._init_weights(self.fc)

    def forward(self, state):
        # 构建mask
        # state: batch, MAX_WORD_LEN
        mask = (state != STATE_MAPPING_STR_TO_INT["PAD"]).float()
        length = mask.sum(1)            # batch,

        x = self.embedding(state)           # [batch, MAX_WORD_LEN, embedding_dim]
        for i, gru in enumerate(self.grus):
            x, _ = gru(x)                   # [batch, MAX_WORD_LEN, hidden_dim*2]

        x = x * mask.unsqueeze(2).expand_as(x)      # [batch, MAX_WORD_LEN, hidden_dim*2]
        x = x.sum(1)                                # [batch, hidden_dim*2]
        x = x / length.unsqueeze(1).expand_as(x)    # [batch, hidden_dim*2]
        x = self.fc(x)                              # [batch, 26]
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

        elif isinstance(module, nn.Embedding):
            padding_idx = 1
            module.weight.data.normal_(mean=0.0, std=0.1)
            module.weight.data[padding_idx].zero_()



class BiGRU(nn.Module):
    def __init__(self,
                 input_dim: int = MAX_WORD_LEN,
                 output_dim: int = OUTPUT_DIM,
                 embedding_dim: int = EMBEDDING_DIM):
        super(BiGRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_sizes = [128, 256, 512, 1024]

        self.embedding = nn.Embedding(28, self.embedding_dim)
        self._init_weights(self.embedding)

        self.grus = nn.ModuleList()
        for i in range(len(self.hidden_sizes)):
            input_dim = self.embedding_dim if i == 0 else self.hidden_sizes[i - 1] * 2
            self.grus.append(
                nn.GRU(input_dim, self.hidden_sizes[i],
                       num_layers=1, bidirectional=True, batch_first=True, bias=True)
            )
        for m in self.grus:
            self._init_weights(m)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1]*2, 1024),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 256),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 26)
        )
        self._init_weights(self.fc)

    def forward(self, state):
        # 构建mask
        # state: batch, MAX_WORD_LEN
        mask = (state != STATE_MAPPING_STR_TO_INT["PAD"]).float()
        length = mask.sum(1)            # batch,

        x = self.embedding(state)           # [batch, MAX_WORD_LEN, embedding_dim]
        for i, gru in enumerate(self.grus):
            x, _ = gru(x)                   # [batch, MAX_WORD_LEN, hidden_dim*2]

        x = x * mask.unsqueeze(2).expand_as(x)      # [batch, MAX_WORD_LEN, hidden_dim*2]
        x = x.sum(1)                                # [batch, hidden_dim*2]
        x = x / length.unsqueeze(1).expand_as(x)    # [batch, hidden_dim*2]
        x = self.fc(x)                              # [batch, 26]
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

        elif isinstance(module, nn.Embedding):
            padding_idx = 1
            module.weight.data.normal_(mean=0.0, std=0.1)
            module.weight.data[padding_idx].zero_()


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


if __name__ == "__main__":
    model = BiGRU()

    state = torch.randint(0, 26, (16, 32))
    print(model(state).shape)

