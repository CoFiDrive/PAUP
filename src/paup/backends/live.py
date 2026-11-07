"""Causal five-frame PAUP fine-tuning around an independently installed frontend.

The external pipeline supplies encoding, learned feature fusion and observed-map
alignment. PAUP requests, scores, bounded masks and the planner loss stay here.
"""
import importlib
import torch
from torch import nn
from torch.utils.data import DataLoader
from paup.models.planner import WaypointPlanner
from paup.models.communication import PAUPCommunication
from paup.requests import gaussian_request
from paup.selection import DEFAULTS,aligned_request
from paup.training import TrainingTask


def factory(name):
    module,attribute=name.split(':',1)
    return getattr(importlib.import_module(module),attribute)


class LivePlanner(nn.Module):
    def __init__(self,frontend,algorithm,feature_channels=128):
        super().__init__()
        self.frontend=frontend.eval().requires_grad_(False)
        self.planner=WaypointPlanner(feature_channels)
        self.communication=PAUPCommunication(algorithm['uncertainty'],
            threshold=algorithm.get('communication',{}).get('threshold',.05),parameters=algorithm.get('selection'))
        self.request_parameters=dict(DEFAULTS,**algorithm.get('selection',{}))

    def load_compatible_state(self,state):
        if any(k.startswith('frontend.') for k in state):return self.load_state_dict(state,strict=True)
        return self.planner.load_compatible_state(state)

    def forward(self,window):
        if not 1<=len(window)<=5:raise ValueError('A causal window must contain1..5 observed frames')
        self.frontend.eval();training=self.planner.training;records=[];previous=None
        times=[float(f['time_ms']) for f in window]
        if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Window times must strictly increase')
        try:
            self.planner.eval()
            with torch.no_grad():
                for frame in window:
                    # Only observed sensor data reaches the frozen encoder.
                    encoded=self.frontend.encode(frame['observation'])
                    request=None
                    if previous is not None:
                        points,info=aligned_request(previous[0],previous[1],frame['pose'],
                            frame['time_ms']-previous[2],self.request_parameters)
                        if points is not None:
                            request=gaussian_request(points[0],height=encoded['predicted'].shape[-2],
                                width=encoded['predicted'].shape[-1],sampling_affines=encoded['request_sampling_affines'])
                    selected=self.communication(encoded['predicted'],encoded['anchors'],encoded['aoi_ms'],request,
                        encoded['history_available'],encoded['horizon_valid'])
                    records.append(self.frontend.fuse(encoded,selected['mask'],frame))
                    inputs=self.frontend.align_history(records)
                    if set(inputs)!={'features','occupancy','target'}:
                        raise ValueError('Planner input builder may return only observed features/maps/navigation target')
                    predicted=self.planner(**inputs)
                    if not torch.isfinite(predicted).all():raise FloatingPointError('Non-finite draft plan')
                    previous=(predicted.detach(),frame['pose'],frame['time_ms'])
        finally:self.planner.train(training)
        # Feedback drafts are detached; the actual final planner pass has gradients.
        return self.planner(**inputs)


def single_window(batch):
    if len(batch)!=1:raise ValueError('Live causal training requires batch size1')
    return batch[0]


def create_task(config):
    b=config['backend'];t=config['training']
    if t['batch_size']!=1:raise ValueError('Live causal training currently requires batch size1')
    frontend=factory(b['frontend_factory'])(b)
    train=factory(b['dataset_factory'])(b,split='training')
    validation=factory(b['dataset_factory'])(b,split='validation')
    algorithm=config['algorithm']
    model=LivePlanner(frontend,algorithm,b.get('feature_channels',128))
    def loss(model,batch):
        # Future GT is used only here, never in window rollout or requests.
        prediction=model(batch['window'])
        target=batch['future_waypoints'].reshape(1,10,2).to(prediction)
        return torch.nn.functional.l1_loss(prediction,target)
    args=dict(batch_size=1,num_workers=t.get('workers',0),collate_fn=single_window)
    return TrainingTask(model,DataLoader(train,shuffle=True,**args),DataLoader(validation,shuffle=False,**args),loss)
