
from collections import deque, namedtuple
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority=1):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super().__init__(action_size, buffer_size, batch_size, seed)

    def add(self, state, action, reward, next_state, done, priority=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)



class ReplayBufferPrioritized(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        super().__init__(action_size, buffer_size, batch_size, seed)
        self.priority = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done, priority=1):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priority.append(priority)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probability = np.array(list(self.priority)) / sum(self.priority)
        indexes = np.random.choice(len(self.memory), self.batch_size, replace=False, p=probability)
        experiences = [self.memory[index] for index in indexes]
        #experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)


