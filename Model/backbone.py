import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from utils.const import *


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class ConvDuelingDQN(nn.Module):

    def __init__(self,
                 input_dim: tuple = (1, MAX_TIME_STEP, MAX_WORD_LEN),
                 output_dim: int = OUTPUT_DIM,
                 embedding_dim: int = EMBEDDING_DIM
                 ):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(28, embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_input_dim = self.feature_size()

        self.value_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, self.output_dim)
        )

    def forward(self, state):
        # state: batch, 1, MAX_TIME_STEP, MAX_WORD_LEN
        state = state.squeeze(1)            # batch, MAX_TIME_STEP, MAX_WORD_LEN
        state = self.embedding(state)       # batch, MAX_TIME_STEP, MAX_WORD_LEN, EMBEDDING_DIM
        state = state.permute(0, 3, 1, 2)   # batch, EMBEDDING_DIM, MAX_TIME_STEP, MAX_WORD_LEN

        features = self.conv(state)
        features = features.reshape(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(
            autograd.Variable(torch.zeros(1, *(EMBEDDING_DIM, self.input_dim[1], self.input_dim[2])))
        ).view(1, -1).size(1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class FCDuelingDQN(nn.Module):

    def __init__(self,
                 input_dim: tuple = (1, MAX_TIME_STEP, MAX_WORD_LEN),
                 output_dim: int = OUTPUT_DIM,
                 embedding_dim: int = EMBEDDING_DIM
                 ):
        super(FCDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(28, embedding_dim)

        self.common_fc = nn.Sequential(
            nn.Linear(input_dim[-1] * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256 * input_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256 * input_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        # state: batch, 1, MAX_TIME_STEP, MAX_WORD_LEN
        state = state.squeeze(1)            # batch, MAX_TIME_STEP, MAX_WORD_LEN
        state = self.embedding(state)       # batch, MAX_TIME_STEP, MAX_WORD_LEN, EMBEDDING_DIM
        state = state.view(state.size(0),
                           state.size(1),
                           -1)              # batch, MAX_TIME_STEP, MAX_WORD_LEN * EMBEDDING_DIM

        features = self.common_fc(state)                        # batch, MAX_TIME_STEP, 256
        features = features.view(features.size(0), -1)          # batch, MAX_TIME_STEP * 256
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.01)


if __name__ == "__main__":
    data = torch.randn((16, 1, MAX_TIME_STEP, MAX_WORD_LEN))
    print(data.shape)       # torch.Size([16, 1, 12, 32])

    net = ConvDuelingDQN()
    out = net(data)
    print(out.shape)        # torch.Size([16, 26])
