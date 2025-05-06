from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass

class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        while self.milestones and self.step_count >= self.milestones[0]:
            self.optimizer.init_lr *= self.gamma
            self.milestones.pop(0)

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.99):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma
