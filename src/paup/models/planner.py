"""Five-observation spatiotemporal waypoint planner, expressed in native PyTorch.

Architecture-compatible with the 128-channel planner used by PAUP. No external
source implementation or parameter tensors are included in this package.
"""
import torch
from torch import nn


def spatial(in_channels,out_channels,stride=1):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride=stride,padding=1),
                         nn.BatchNorm2d(out_channels),nn.ReLU())


class TemporalStage(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.spatial=nn.Sequential(spatial(in_channels,out_channels,2),spatial(out_channels,out_channels))
        self.temporal=nn.Sequential(nn.Conv3d(out_channels,out_channels,(3,1,1)),
                                   nn.BatchNorm3d(out_channels),nn.ReLU())

    def forward(self,x):
        b,t,c,h,w=x.shape
        x=self.spatial(x.reshape(b*t,c,h,w))
        x=x.reshape(b,t,*x.shape[1:]).permute(0,2,1,3,4)
        return self.temporal(x).permute(0,2,1,3,4).contiguous()


class WaypointPlanner(nn.Module):
    def __init__(self,feature_channels=128):
        super().__init__()
        self.feature_channels=feature_channels
        self.map_embed=nn.Sequential(spatial(5,32),spatial(32,32))
        self.feature_embed=nn.Sequential(spatial(feature_channels,32),spatial(32,32))
        self.merge=nn.Sequential(spatial(64,32),spatial(32,32))
        self.stages=nn.ModuleList([TemporalStage(32,64),TemporalStage(64,128)])
        self.final=spatial(128,256,2)
        self.target=nn.Sequential(nn.Linear(2,16),nn.Linear(16,64),nn.Linear(64,128))
        self.decoder=nn.Sequential(nn.Linear(384,1025),nn.Linear(1025,512),nn.Linear(512,20))

    def forward(self,features,occupancy,target):
        if features.ndim!=5 or features.shape[1]!=5 or features.shape[2]!=self.feature_channels:
            raise ValueError('Expected B,5,C,H,W aligned feature history')
        if occupancy.shape[:3]!=(features.shape[0],5,5) or occupancy.shape[-2:]!=features.shape[-2:]:
            raise ValueError('Expected matching B,5,5,H,W observed map channels')
        b,t,c,h,w=features.shape
        xf=self.feature_embed(features.reshape(b*t,c,h,w))
        xm=self.map_embed(occupancy.reshape(b*t,5,h,w))
        x=self.merge(torch.cat([xm,xf],1)).reshape(b,t,32,h,w)
        for stage in self.stages:x=stage(x)
        x=self.final(x[:,0]).mean((-2,-1))
        return self.decoder(torch.cat([x,self.target(target)],1)).reshape(b,10,2)

    def load_compatible_state(self,state):
        """Strictly load a private legacy planner state; nothing is bundled/downloaded."""
        state={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        if set(state)==set(self.state_dict()):return self.load_state_dict(state,strict=True)
        mapping={}
        for ours,prefix,suffix in [('map_embed','', '_no_occ'),('feature_embed','_f',''),('merge','_f2','')]:
            for layer in [1,2]:
                conv='conv_pre_'+str(layer)+prefix+(suffix if layer==1 else '')
                bn='bn_pre_'+str(layer)+prefix
                mapping[f'{ours}.{layer-1}.0']=conv;mapping[f'{ours}.{layer-1}.1']=bn
        for i in [0,1]:
            for j in [0,1]:
                mapping[f'stages.{i}.spatial.{j}.0']=f'conv{i+1}_{j+1}'
                mapping[f'stages.{i}.spatial.{j}.1']=f'bn{i+1}_{j+1}'
            mapping[f'stages.{i}.temporal.0']=f'conv3d_{i+1}.conv3d'
            mapping[f'stages.{i}.temporal.1']=f'conv3d_{i+1}.bn3d'
        mapping['final.0']='conv3_1';mapping['final.1']='bn3_1'
        for j in range(3):mapping[f'target.{j}']=f'target_encoder.layers.{j}';mapping[f'decoder.{j}']=f'decoder.layers.{j}'
        converted={}
        for key in self.state_dict():
            prefix,field=key.rsplit('.',1);converted[key]=state[mapping[prefix]+'.'+field]
        consumed={mapping[k.rsplit('.',1)[0]]+'.'+k.rsplit('.',1)[1] for k in converted}
        if set(state)!=consumed:raise ValueError('Unexpected planner checkpoint fields')
        return self.load_state_dict(converted,strict=True)
