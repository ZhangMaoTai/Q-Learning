import json

import numpy as np
import pandas as pd

import torch

from utils.const import *


def padding(word_list,
            max_len):
    n = len(word_list)

    if n >= max_len:
        return word_list[0:max_len]
    else:
        return word_list + [STATE_MAPPING_STR_TO_INT["PAD"]] * (max_len - n)


class Model_Data(torch.utils.data.Dataset):
    def __init__(self,
                 json_path,
                 max_len: int = MAX_WORD_LEN):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.x = self.data["x"]
        self.y = self.data["y"]
        self.max_len = max_len

        self.n = len(self.y)

        self.x_torch = []
        self.y_torch = []

        for i in range(self.n):
            word = [STATE_MAPPING_STR_TO_INT[letter] for letter in self.x[i]]
            word = padding(word, self.max_len)
            self.x_torch.append(
                torch.tensor(word, dtype=torch.int64)
            )                                                       # shape = [max_lan]

            index = [ACTION_MAPPING_STR_TO_INT[letter] for letter in self.y[i]]
            y = torch.zeros(26, dtype=torch.float32)
            y[index] = 1
            self.y_torch.append(y)                                  # shape = [26]

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.x_torch[item], self.y_torch[item]


if __name__ == "__main__":
    dataset = Model_Data("../test.json")
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=8,
                                             shuffle=True)

    a = iter(dataloader)
    x, y = a.next()
    print(x)
    print(x.shape)

    print(y)            # torch.Size([8, 32])
    print(y.shape)      # torch.Size([8, 26])
