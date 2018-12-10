import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from replay_buffer import SimpleReplayBuffer, ReplayBufferPrioritized
from noise import OUNoise, RandomNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

NOISE_DISCRIMINATION = 0.999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, number_of_independednt_agents, random_seed, enable_priority=False, BUFFER_SIZE=int(1e5), BATCH_SIZE=128, GAMMA=0.99, TAU=1e-3, LR_ACTOR=1e-5, LR_CRITIC=1e-4, WEIGHT_DECAY=0, index_of_agent_in_maddpg=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.number_of_agents = number_of_independednt_agents
        self.seed = random.seed(random_seed)
        '''just for multiagent mode: index of agent's data in state/action etc'''
        self.agent_index = index_of_agent_in_maddpg

        self.max_priority_value = 0

        self.UPDATE_STEPS = 3
        self.MIN_PRIORITY = 0.2
        self.BUFFER_SIZE = BUFFER_SIZE  # replay buffer size
        self.BATCH_SIZE = BATCH_SIZE  # minibatch size
        self.GAMMA = GAMMA  # discount factor
        self.TAU = TAU  # for soft update of target parameters
        self.LR_ACTOR = LR_ACTOR  # learning rate of the actor
        self.LR_CRITIC = LR_CRITIC  # learning rate of the critic
        self.WEIGHT_DECAY = WEIGHT_DECAY  # L2 weight decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * number_of_independednt_agents,
                                   action_size * number_of_independednt_agents,
                                   random_seed).to(device)
        self.critic_target = Critic(state_size * number_of_independednt_agents,
                                    action_size * number_of_independednt_agents,
                                    random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        # Noise process
        #self.noise = OUNoise(action_size, random_seed, sigma=1.0)
        self.noise = RandomNoise(action_size, sigma=1.0)

        # Replay memory
        if enable_priority:
            self.memory = ReplayBufferPrioritized(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        else:
            self.memory = SimpleReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        #if self.agent_index is not(None):
        #    state = state[self.agent_index]
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:

            if self.noise.sigma < 0.4:
                action += self.noise.sample()
            else:
                action = self.noise.sample()
            self.noise.sigma = max(self.noise.sigma * NOISE_DISCRIMINATION, 0.1)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        raise NotImplementedError()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_priority(self, s, a, r, ns, d):
        raise NotImplementedError()

    def updateNetworks(self):
        #sself.critic_target.load_state_dict(torch.load('checkpoint_critic.pth'))
        self.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
        #self.actor_target.load_state_dict(torch.load('checkpoint_actor.pth'))
        self.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))


class myDDPG(Agent):
        def __init__(self, state_size, action_size, random_seed, enable_priority=False,
                 BUFFER_SIZE=int(1e5),
                 BATCH_SIZE=128, GAMMA=0.99, TAU=1e-3, LR_ACTOR=1e-5, LR_CRITIC=1e-4, WEIGHT_DECAY=0,
                 index_of_agent_in_maddpg=None):
            super().__init__(state_size, action_size, 1, random_seed,
                             enable_priority, BUFFER_SIZE, BATCH_SIZE,
                             GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY)

        def step(self, states, actions, rewards, next_states, dones):
            """ agent_index is only used for MADDPG - it is index of current agent in other input arguments"""
            """Save experience in replay memory, and use random sample from buffer to learn."""
            # Save experience / reward

            for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                error = self.get_priority(s, a, r, ns, d)
                self.memory.add(s, a, r, ns, d, priority=error)

            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.UPDATE_STEPS):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.GAMMA)

        def learn(self, experiences, gamma):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, self.TAU)
            self.soft_update(self.actor_local, self.actor_target, self.TAU)

        def get_priority(self, s, a, r, ns, d):
            if isinstance(self.memory, ReplayBufferPrioritized):
                next_state_cp = np.copy(ns)
                state_cp = np.copy(s)
                action_cp = np.copy(a)

                next_state_cp.resize(1, self.state_size)
                state_cp.resize(1, self.state_size)
                action_cp.resize(1, self.action_size)

                next_state_tourch = torch.from_numpy(next_state_cp).float().to(device)
                action_next = self.actor_target(next_state_tourch)
                Q_targets_next = self.critic_target(next_state_tourch, action_next)
                Q_targets = torch.tensor(r) + (self.GAMMA * Q_targets_next * (1 - d))

                state_tourch = torch.from_numpy(state_cp).float().to(device)
                action_tourch = torch.from_numpy(action_cp).float().to(device)
                Q_expected = self.critic_local(state_tourch, action_tourch)
                priority = abs((Q_expected - Q_targets).item())
                priority += self.MIN_PRIORITY
            else:
                priority = [1 for _ in range(a.size)]

            if (self.max_priority_value < priority):
                print('max_priority: {:.4f}'.format(priority))
                self.max_priority_value = priority
            return priority