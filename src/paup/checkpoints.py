"""Load user-supplied trusted training checkpoints across supported Torch versions."""
import inspect
import torch


def load_checkpoint(path):
    kwargs={'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only']=False
    return torch.load(path,**kwargs)
