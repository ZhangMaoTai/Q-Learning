import os
import numpy as np
from tqdm.auto import tqdm
import wandb
from utils.util import wandb_start
from QLearn.experience import BasicBuffer
from QLearn.environment import environment
from Model.models import Agent


class QTrainer:
    def __init__(self,
                 env: environment,
                 agent: Agent,
                 replay_buffer: BasicBuffer,
                 mini_epoch
                 ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.agent = agent

        self.mini_epoch = mini_epoch

        # init wandb
        self.writer = wandb_start(True)

    def train(self,
              num_updates,
              save_dir):

        for updates in range(1, num_updates+1):
            num_episode = 0
            num_word_success = 0
            num_letter_success = 0
            num_letter_play = 0
            episode_reward_list = []
            episode_len_list = []

            # collect data
            self.agent.eval()
            progress_bar = tqdm(range(self.replay_buffer.max_size))
            while True:
                # each episode
                state = self.env.reset()
                episode_reward = 0
                episode_len = 0
                false_action = []

                while True:
                    action = self.agent.get_action(state=state,
                                                   false_action=false_action)
                    next_state, new_word, reward, done, word_success, letter_success = self.env.step(action)
                    exp_dataset = self.replay_buffer.push(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    num_letter_success += (1 if letter_success else 0)
                    num_letter_play += 1
                    episode_len += 1

                    if not letter_success:
                        false_action.append(action)
                    progress_bar.update(1)

                    if done or exp_dataset is not None:
                        num_episode += 1
                        num_word_success += (1 if word_success else 0)
                        episode_reward_list.append(episode_reward)
                        episode_len_list.append(episode_len)
                        self.agent.reset_noise()
                        break

                if exp_dataset is not None:
                    self.writer.add_scalar("losses/word_success", num_word_success / num_episode, updates)
                    self.writer.add_scalar("losses/letter_success", num_letter_success / num_letter_play, updates)
                    self.writer.add_scalar("losses/rewards", np.sum(episode_reward_list) / num_episode, updates)
                    self.writer.add_scalar("losses/episode_len", np.mean(episode_len_list), updates)
                    break

            # train
            if exp_dataset is not None:
                self.agent.train()
                progress_bar = tqdm(range(self.mini_epoch))

                for _ in range(self.mini_epoch):
                    for batch in exp_dataset:
                        loss = self.agent.compute_loss(batch)

                        self.agent.optimizer.zero_grad()
                        loss.backward()
                        self.agent.optimizer.step()

                        self.agent.update_target()

                        self.writer.add_scalar("losses/loss", loss, updates)

                    progress_bar.update(1)

            if updates % 1000 == 0:
                self.agent.save_checkpoint(
                    os.path.join(save_dir, "{}update.pth".format(updates))
                )

        self.writer.close()
        wandb.finish()





