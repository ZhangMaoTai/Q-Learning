import random

import torch
import torch.nn as nn
import torch.autograd as autograd
from transformers import get_scheduler
import numpy as np

from QLearn.experience import BasicBuffer
from Model.backbone import ConvDuelingDQN, init_weights, FCDuelingDQN


class Agent:

    def __init__(self,
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.99,
                 num_warmup_steps=100,
                 num_training_steps=1000
                 ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_model = FCDuelingDQN().to(self.device)
        self.target_model = FCDuelingDQN().to(self.device)

        self.eval_model.apply(init_weights)
        self.target_model.apply(init_weights)

        # hard copy model parameters to target model parameters
        for eval_param, target_param in zip(self.eval_model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(eval_param)

        self.optimizer = torch.optim.AdamW(self.eval_model.parameters(),
                                           lr=self.learning_rate)
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        self.MSE_loss = nn.MSELoss()

    def get_action(self,
                   state,
                   history_action: list,
                   eps=0.20):
        state = state.to(self.device)
        qvals = self.eval_model(state).cpu().detach().numpy()

        qvals[0][history_action] = float('-inf')      # 不考虑history_action
        action = np.argmax(qvals)

        if np.random.randn() > eps:
            return random.sample(list(range(26)), 1)[0]
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)                 # torch.Size([batch, 1, 12, 32])
        actions = actions.to(self.device)               # torch.Size([batch])
        rewards = rewards.to(self.device)               # torch.Size([batch])
        next_states = next_states.to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)    # torch.Size([batch])

        curr_Q = self.eval_model(states).gather(1, actions.unsqueeze(1))         # torch.Size([batch, 1])
        curr_Q = curr_Q.squeeze(1)                      # torch.Size([batch])

        # DQN
        next_Q = self.target_model(next_states).detach()
        max_next_Q = torch.max(next_Q, 1)[0]          # torch.Size([batch])

        # # DDQN
        # with torch.no_grad():
        #     next_a = self.eval_model(next_states).argmax(-1)     # torch.Size([batch])
        #     max_next_Q = self.target_model(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)   # torch.Size([batch])

        expected_Q = rewards + self.gamma * max_next_Q * (1. - dones)

        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update_target(self):
        for q_target_params, q_eval_params in zip(self.target_model.parameters(),
                                                  self.eval_model.parameters()):
            q_target_params.data.copy_(self.tau * q_eval_params + (1 - self.tau) * q_target_params)

    def reset_noise(self):
        self.eval_model.reset_noise()
        self.target_model.reset_noise()

    def train(self):
        self.eval_model.train()
        self.target_model.train()

    def eval(self):
        self.eval_model.eval()
        self.target_model.eval()

    def save_checkpoint(self, path):
        ckp = {
            "model": self.eval_model.state_dict(),
        }
        torch.save(ckp, path)


if __name__ == "__main__":
    from utils.const import *
    agent = Agent()

    state = torch.randint(0, 26, (1, 1, MAX_TIME_STEP, MAX_WORD_LEN))
    print(agent.get_action(state, []))

    states = torch.randint(0, 26, (16, 1, MAX_TIME_STEP, MAX_WORD_LEN))
    next_states = torch.randint(0, 26, (16, 1, MAX_TIME_STEP, MAX_WORD_LEN))
    actions = torch.randint(0, 26, (16,))
    rewards = torch.randn((16,))
    dones = [True] * 16

    loss = agent.compute_loss((states, actions, rewards, next_states, dones))
    print(loss)
