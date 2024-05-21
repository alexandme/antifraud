from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor
import copy
import torch
import os

class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0, save_path='./best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        self.save_path = save_path  # Path to save the best model

    def earlystop(self, loss, model=None):
        value = -loss
        cv = loss
        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            torch.save(self.best_model.state_dict(), self.save_path)  # Save the best model
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            torch.save(self.best_model.state_dict(), self.save_path)  # Save the best model
            self.count = 0
