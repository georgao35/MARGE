import math
from abc import ABC, abstractmethod


class _LRScheduler(ABC):
    def __init__(self, initial_lr, total_steps):
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.lr_history = []
        self.current_lr = initial_lr
        
    def step(self):
        self.current_step += 1
        self.current_lr = self._get_lr()
        self.lr_history.append(self.current_lr)
        return self.current_lr
    
    @abstractmethod
    def _get_lr(self):
        pass
    
    def get_lr_history(self):
        return self.lr_history


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, initial_lr, total_steps, min_lr=0):
        super().__init__(initial_lr, total_steps)
        self.min_lr = min_lr
        
    def _get_lr(self):
        progress = self.current_step / self.total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class LinearAnnealingLR(_LRScheduler):
    def __init__(self, initial_lr, total_steps, end_lr=0):
        super().__init__(initial_lr, total_steps)
        self.end_lr = end_lr
        
    def _get_lr(self):
        progress = self.current_step / self.total_steps
        return self.end_lr + (self.initial_lr - self.end_lr) * (1 - progress)


class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, initial_lr, total_steps, warmup_steps, min_lr=0):
        super().__init__(initial_lr, total_steps)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Ensure progress doesn't exceed 1
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_lr_schedule(scheduler, title="Learning Rate Schedule"):
        # Run the scheduler for all steps
        lrs = []
        for _ in range(scheduler.total_steps):
            lr = scheduler.step()
            lrs.append(lr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(scheduler.total_steps), lrs)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.show()

    # Example usage with different schedulers
    def compare_schedulers():
        # Create different schedulers
        cosine = CosineAnnealingLR(initial_lr=0.1, total_steps=100, min_lr=0.001)
        linear = LinearAnnealingLR(initial_lr=0.1, total_steps=100, end_lr=0.001)
        cosine_warmup = CosineAnnealingWarmupLR(
            initial_lr=0.1, 
            total_steps=100, 
            warmup_steps=10, 
            min_lr=0.001
        )
        
        # Plot each scheduler
        plt.figure(figsize=(12, 8))
        
        # Run each scheduler and plot
        schedulers = {
            'Cosine Annealing': cosine,
            'Linear Annealing': linear,
            'Cosine Annealing with Warmup': cosine_warmup
        }
        
        for name, scheduler in schedulers.items():
            lrs = []
            for _ in range(scheduler.total_steps+20):
                lr = scheduler.step()
                lrs.append(lr)
            plt.plot(range(scheduler.total_steps+20), lrs, label=name)
        
        plt.title('Learning Rate Schedules Comparison')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig("test1.png")

    # Example usage:
    # Plot single scheduler
    scheduler = CosineAnnealingWarmupLR(
        initial_lr=0.1,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.001
    )
    plot_lr_schedule(scheduler, "Cosine Annealing with Warmup")

    # Compare different schedulers
    compare_schedulers()
