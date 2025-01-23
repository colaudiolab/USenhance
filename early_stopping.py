import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            if metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric > self.best_metric + self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
