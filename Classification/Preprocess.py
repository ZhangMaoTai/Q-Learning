import numpy as np
import random
import json
from itertools import combinations
from utils.const import *


def construct_data(txt_path="../words_240000.txt"):
    text_file = open(txt_path, "r")
    data = text_file.read().splitlines()
    text_file.close()

    train_x = []
    train_y = []

    for word in data:
        unique_letter = set(word)

        # 遍历所有的预测情况
        for r in range(len(unique_letter)):
            combination = combinations(unique_letter, r)        # 0个到len-1个

            # 现在预测出来了这个combine
            for combine in combination:
                x = "".join(
                    [
                        letter if letter in combine else "_"
                        for letter in word
                    ]
                )

                y = list(unique_letter - set(combine))

                train_x.append(x)
                train_y.append(y)

    return train_x, train_y


def save_data(x, y, save_json):
    with open(save_json, "w") as f:
        json.dump(
            {"x": x, "y": y},
            f
        )


if __name__ == "__main__":
    train_x, train_y = construct_data("../words_240000.txt")
    test_x, test_y = construct_data("../words_10000.txt")

    print("train :", len(train_x))
    print("test :", len(test_x))

    save_data(train_x, train_y, "../train.json")
    save_data(test_x, test_y, "../test.json")

    with open("../test.json", "r") as f:
        data = json.load(f)

    with open("../test_small.json", "w") as f:
        json.dump(
            {"x": data["x"][0:10000], "y": data["y"][0:10000]},
            f
        )

    # train: 96483221
    # test: 4242261
