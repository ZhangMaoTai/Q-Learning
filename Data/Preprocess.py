import numpy as np
import random

text_file = open("../words_250000.txt", "r")
data = text_file.read().splitlines()
text_file.close()

random.shuffle(data)

with open("../words_240000.txt", "w") as f_train:
    with open("../words_10000.txt", "w") as f_test:
        for i in range(len(data)):
            if i <= 10000:
                f_test.write(data[i] + '\n')
            else:
                f_train.write(data[i] + '\n')


if __name__ == "__main__":

    text_file = open("/Users/zhangjiasheng/Desktop/交大/Trex/words_250000_train.txt", "r")
    data = text_file.read().splitlines()
    text_file.close()

    print(data[0:10])
    print(np.mean([len(s) for s in data]))
    print(np.max([len(s) for s in data]))
    print(np.min([len(s) for s in data]))

