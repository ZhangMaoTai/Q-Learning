import os
import numpy as np
import torch.nn.utils
from tqdm.auto import tqdm
import wandb
from utils.util import wandb_start
from QLearn.experience import BasicBuffer
from QLearn.environment import environment
from QLearn.eval import Evaler
from Model.models import Agent
from utils.const import *


class QTrainer:
    def __init__(self,
                 train_env: environment,
                 evaler: Evaler,
                 agent: Agent,
                 replay_buffer: BasicBuffer,
                 mini_epoch: int,
                 per_updates_eval: int,
                 log
                 ):
        self.train_env = train_env
        self.evaler = evaler
        self.replay_buffer = replay_buffer
        self.agent = agent
        self.log = log

        self.mini_epoch = mini_epoch
        self.per_updates_eval = per_updates_eval

        # init wandb
        self.writer = wandb_start(True)

    def train(self,
              num_updates,
              save_dir,
              max_norm):

        for updates in range(1, num_updates+1):
            num_episode = 0
            num_word_success = 0
            num_letter_success = 0
            num_letter_play = 0
            episode_reward_list = []
            episode_len_list = []
            log_flag = True

            # collect data

            self.agent.eval()
            # self.agent.train()
            progress_bar = tqdm(range(self.replay_buffer.max_size))
            while True:
                # each episode
                state = self.train_env.reset()
                episode_reward = 0
                episode_len = 0
                history_action = []

                use_baseline = True
                exp_dataset = None
                num_baseline_success = 0

                while True:
                    if use_baseline:
                        action = self.train_env.baseline_model.guess(
                            " ".join(self.train_env.word_list[-1]) + " ",
                            guessed_letters=[ACTION_MAPPING_INT_TO_STR[int(i)] for i in history_action]
                        )
                        action = ACTION_MAPPING_STR_TO_INT[action]
                    else:
                        action = self.agent.get_action(state=state,
                                                       history_action=history_action)

                    next_state, new_word, reward, done, word_success, letter_success = self.train_env.step(action)
                    if letter_success:
                        num_baseline_success += 1
                        if num_baseline_success >= 2:
                            use_baseline = False

                    if not use_baseline:
                        exp_dataset = self.replay_buffer.push(state, action, reward, next_state, done)
                        episode_reward += reward
                        num_letter_success += (1 if letter_success else 0)
                        num_letter_play += 1

                    state = next_state
                    episode_len += 1
                    history_action.append(action)
                    progress_bar.update(1)

                    if done or exp_dataset is not None:
                        num_episode += 1
                        num_word_success += (1 if word_success else 0)
                        episode_reward_list.append(episode_reward)
                        episode_len_list.append(episode_len)
                        self.agent.reset_noise()

                        if log_flag:
                            self.log.info("Current word: {}. History action: {}. Success: {}".format(
                                self.train_env.show_current_word(), " ".join(
                                    [ACTION_MAPPING_INT_TO_STR[int(i)] for i in history_action]
                                ), word_success
                                )
                            )
                            log_flag = False
                        break

                if exp_dataset is not None:
                    self.writer.add_scalar("losses/word_success", num_word_success / num_episode, updates)
                    self.writer.add_scalar("losses/letter_success", num_letter_success / num_letter_play, updates)
                    self.writer.add_scalar("losses/rewards", np.sum(episode_reward_list) / num_episode, updates)
                    self.writer.add_scalar("losses/episode_len", np.sum(episode_len_list) / num_episode, updates)
                    self.writer.add_scalar("losses/num_episode", num_episode, updates)
                    break

            # train
            if exp_dataset is not None:
                self.agent.train()
                progress_bar = tqdm(range(self.mini_epoch))

                for _ in range(self.mini_epoch):
                    for batch in exp_dataset:
                        self.agent.optimizer.zero_grad()

                        loss = self.agent.compute_loss(batch)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.agent.eval_model.parameters(), max_norm)
                        self.agent.optimizer.step()
                        self.agent.scheduler.step()

                        self.agent.update_target()
                        self.writer.add_scalar("losses/loss", loss, updates)

                    progress_bar.update(1)

            # eval & save
            if updates % self.per_updates_eval == 0:
                # eval
                self.evaler.set_agent(self.agent)
                result = self.evaler.eval()
                self.writer.add_scalar("TestSet/word_success_rate", result['word_success_rate'], updates)
                self.writer.add_scalar("TestSet/letter_success_rate", result['letter_success_rate'], updates)
                self.writer.add_scalar("TestSet/episode_reward_mean", result['episode_reward_mean'], updates)
                self.writer.add_scalar("TestSet/episode_len_mean", result['episode_len_mean'], updates)

                # save
                self.agent.save_checkpoint(
                    os.path.join(save_dir, "{}update.pth".format(updates))
                )

        self.writer.close()
        wandb.finish()





