import numpy as np
import random
import copy

class Noise:
    def __init__(self, size, seed, mu=0., theta=0.7, sigma=0.1):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class OUNoise(Noise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.7, sigma=0.1):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class RandomNoise(Noise):
    def __init__(self, size, sigma=0.2):
        self.sigma = sigma
        self.size = size

    def reset(self):
        pass

    def sample(self):
        noise = (np.random.rand(self.size) - 0.5) * 2
        noise *= self.sigma
        return noise