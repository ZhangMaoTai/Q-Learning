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
            num_success = 0
            episode_reward_list = []

            # collect data
            self.agent.eval()
            progress_bar = tqdm(range(self.replay_buffer.max_size))
            while True:
                # each episode
                state = self.env.reset()
                episode_reward = 0

                while True:
                    action = self.agent.get_action(state)
                    next_state, new_word, reward, done, success = self.env.step(action)
                    exp_dataset = self.replay_buffer.push(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    progress_bar.update(1)

                    if done or exp_dataset is not None:
                        num_episode += 1
                        num_success += (1 if success else 0)
                        episode_reward_list.append(episode_reward)
                        self.agent.reset_noise()
                        break

                if exp_dataset is not None:
                    self.writer.add_scalar("losses/success", num_success / num_episode, updates)
                    self.writer.add_scalar("losses/rewards", np.mean(episode_reward_list), updates)
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





