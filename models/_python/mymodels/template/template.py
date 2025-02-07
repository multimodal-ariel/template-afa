import torch as th


def make_model() -> th.nn.Module:
    return th.nn.Sequential(th.nn.Linear(10, 512), th.nn.ReLU(), th.nn.Linear(512, 10))
