import torch
import random
from utils.const import *


def cal_reward(is_match: bool,
               num_try: int,
               remain_try: int):
    if is_match:
        return 1.
    else:
        return -1.


def right_pad(letter_list,
              max_len=MAX_WORD_LEN):
    n = len(letter_list)

    if n >= max_len:
        return letter_list[0:max_len]
    else:
        return letter_list + ["PAD"] * (max_len - n)


def pad_time_step(word_list,
                  max_len=MAX_TIME_STEP):
    n = len(word_list)

    if n >= max_len:
        return word_list[-max_len:]
    else:
        return word_list + [["PAD" for _ in range(len(word_list[0]))]] * (max_len - n)


class environment:
    def __init__(self,
                 vocab_path: str,
                 max_try: int = MAX_TRY):
        self.vocab_path = vocab_path
        self.max_try = max_try

        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab)

        self.current_id = -1
        self.current_word = None
        self.num_try = None
        self.remain_try = None
        self.action_list = None
        self.word_list = None

    def get_vocab(self):
        text_file = open(self.vocab_path, "r")
        data = text_file.read().splitlines()
        text_file.close()

        random.shuffle(data)
        return data

    def show_current_word(self):
        return self.current_word

    def reset(self):
        self.current_id += 1
        self.current_word = self.vocab[self.current_id]
        self.num_try = 0
        self.remain_try = self.max_try
        self.action_list = []
        self.word_list = ["_" * len(self.current_word)]
        return self.get_state()

    def step(self, action: int):
        action_letter = ACTION_MAPPING_INT_TO_STR[action]

        if self.remain_try > 0:
            self.num_try += 1
            self.action_list.append(action_letter)
            last_pred_word = self.word_list[-1]

            match_index = [i
                           for i in range(len(self.current_word))
                           if self.current_word[i] == action_letter]

            if len(match_index) == 0:
                self.remain_try -= 1
                new_word = last_pred_word
                reward = cal_reward(is_match=False, num_try=self.num_try, remain_try=self.remain_try)
            else:
                new_word = [
                    action_letter if i in match_index else last_pred_word[i]
                    for i in range(len(self.current_word))
                ]
                new_word = "".join(new_word)
                reward = cal_reward(is_match=True, num_try=self.num_try, remain_try=self.remain_try)

            self.word_list.append(new_word)

            if "_" not in new_word or self.remain_try == 0:
                done = True
                if "_" not in new_word:
                    success = True
                else:
                    success = False

            else:
                done = False
                success = None

            return self.get_state(), new_word, reward, done, success

        else:
            raise RuntimeError("Can't call step because it is over.")

    def get_state(self):
        word_list = [
            right_pad(list(word), MAX_WORD_LEN)
            for word in self.word_list
        ]                                                                   # list[list[str]]
        word_list = pad_time_step(word_list, MAX_TIME_STEP)                 # list[list[str]]

        id_list = [
            [STATE_MAPPING_STR_TO_INT[letter] for letter in word]
            for word in word_list
        ]                                                                   # list[list[int]]

        return torch.tensor(id_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)      # shape = (1, 1, MAX_TIME_STEP, MAX_WORD_LEN)


if __name__ == "__main__":
    env = environment("/Users/zhangjiasheng/Desktop/交大/Trex/words_250000_train.txt")
    print(env.show_current_word())

    round = 0
    while True:
        state, new_word, reward, done = env.step(1)
        print(new_word, reward, done)
        print(state.shape)

        if done:
            env.reset()
            round += 1
        if round == 2:
            break


