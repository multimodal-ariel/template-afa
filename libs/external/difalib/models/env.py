from copy import deepcopy

import torch


class Env(object):
    def __init__(self, params, model):
        self.params = deepcopy(params)
        self.model = model

    def reset(self):
        pass

    def step(self):
        pass
