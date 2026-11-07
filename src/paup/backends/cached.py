"""Planner training from fixed, observed, pose-aligned perception feature caches."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from paup.models.planner import WaypointPlanner
from paup.training import TrainingTask


class FeatureWindows(Dataset):
    def __init__(self,manifest):
        self.manifest=Path(manifest)
        spec=json.loads(self.manifest.read_text())
        if spec.get('schema')!='paup_feature_windows_v1':raise ValueError('Unsupported feature-window manifest')
        if spec.get('occupancy_source')!='observations':
            raise ValueError('Occupancy must come from observations / perception, not future GT')
        if not spec.get('pose_aligned',False):raise ValueError('Pose-align the history before caching')
        self.windows=[(self.manifest.parent/filename).resolve() for filename in spec['windows']]
        if not self.windows:raise ValueError('Empty training manifest')
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        with np.load(self.windows[index],allow_pickle=False) as archive:
            batch={k:torch.from_numpy(archive[k].astype(np.float32,copy=True)) for k in ['features','occupancy','target','future_waypoints']}
        if batch['features'].ndim!=4 or batch['features'].shape[0]!=5:
            raise ValueError('Feature window must have shape5,C,H,W')
        if batch['occupancy'].shape!=(5,5,*batch['features'].shape[-2:]):raise ValueError('Invalid observation-map shape')
        if batch['target'].shape!=(2,) or batch['future_waypoints'].shape!=(10,2):raise ValueError('Invalid target shape')
        if not all(torch.isfinite(v).all() for v in batch.values()):raise ValueError('Non-finite training example')
        return batch


def create_task(config):
    b=config['backend'];t=config['training']
    train=FeatureWindows(b['train_manifest']);val=FeatureWindows(b['validation_manifest'])
    overlap=set(train.windows)&set(val.windows)
    if overlap:raise ValueError('Training and validation windows overlap')
    model=WaypointPlanner(b.get('feature_channels',128))
    def loss(model,batch):
        prediction=model(batch['features'],batch['occupancy'],batch['target'])
        return torch.nn.functional.l1_loss(prediction,batch['future_waypoints'])
    kwargs=dict(batch_size=t['batch_size'],num_workers=t.get('workers',4),pin_memory=torch.cuda.is_available())
    return TrainingTask(model,DataLoader(train,shuffle=True,**kwargs),DataLoader(val,shuffle=False,**kwargs),loss)
