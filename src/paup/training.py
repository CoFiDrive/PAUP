"""Shared optimizer loop for training-only backends."""
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class TrainingTask:
    model: torch.nn.Module
    train_loader: object
    validation_loader: object
    loss: Callable


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


def epoch(task,loader,device,optimizer=None,clip_grad=10.):
    task.model.train(optimizer is not None)
    total=0.;count=0
    for batch in loader:
        if batch is None:continue
        batch=move(batch,device)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss=task.loss(task.model,batch)
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError('Training backend returned a non-finite or non-scalar loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in task.model.parameters() if p.requires_grad],
                                              clip_grad,error_if_nonfinite=True)
                optimizer.step()
        total+=float(loss.detach());count+=1
    if count==0:raise RuntimeError('No usable batches; check dataset split and collation')
    return total/count
