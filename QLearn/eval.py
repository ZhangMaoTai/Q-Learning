import numpy as np
from tqdm.auto import tqdm
from QLearn.environment import environment
from Model.models import Agent
from utils.const import *


class Evaler:
    def __init__(self,
                 env: environment,
                 agent: Agent):
        self.env = env
        self.agent = agent
        self.agent.eval()

    def eval(self):
        num_episode = 0
        num_word_success = 0
        num_letter_success = 0
        num_letter_play = 0
        episode_reward_list = []
        episode_len_list = []
        episode_action_list = []
        episode_word_list = []

        # collect data
        self.env.from_start()

        progress_bar = tqdm(range(self.env.vocab_size))
        for _ in range(self.env.vocab_size):
            # each episode
            state = self.env.reset()
            episode_reward = 0
            episode_len = 0
            history_action = []
            episode_word_list.append(self.env.show_current_word())

            while True:
                action = self.agent.get_action(state=state,
                                               history_action=history_action)
                next_state, new_word, reward, done, word_success, letter_success = self.env.step(action)

                state = next_state
                episode_reward += reward
                num_letter_success += (1 if letter_success else 0)
                num_letter_play += 1
                episode_len += 1

                history_action.append(action)

                if done:
                    num_episode += 1
                    num_word_success += (1 if word_success else 0)
                    episode_reward_list.append(episode_reward)
                    episode_len_list.append(episode_len)
                    episode_action_list.append(
                        " ".join([ACTION_MAPPING_INT_TO_STR[int(i)] for i in history_action])
                    )
                    progress_bar.update(1)
                    break
        return {
            "num_episode": num_episode,
            "word_success_rate": num_word_success / num_episode,
            "letter_success_rate": num_letter_success / num_letter_play,

            "episode_reward_list": episode_reward_list,
            "episode_reward_mean": np.mean(episode_reward_list),

            "episode_len_list": episode_len_list,
            "episode_len_mean": np.mean(episode_len_list),

            "episode_action_list": episode_action_list,
            "episode_word_list": episode_word_list
        }

    def set_agent(self, agent):
        self.agent = agent
        self.agent.eval()


if __name__ == "__main__":
    evaler = Evaler(
        agent=Agent(),
        env=environment(vocab_path="C:\\CODE学习\\GitRepo\\Q-Learning\\words_10000.txt"))

    result = evaler.eval()

    print(result["num_episode"])
    print(result["word_success_rate"])
    print(result["letter_success_rate"])

    print(result["episode_reward_mean"])
    print(result["episode_len_mean"])
    print(np.max(result["episode_len_list"]))

    print(result["episode_action_list"][0:10])
    print(result["episode_word_list"][0:10])
