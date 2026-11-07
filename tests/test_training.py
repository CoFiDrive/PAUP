import json
import numpy as np
import torch
import yaml
from paup.train import main
from paup.models.planner import WaypointPlanner
from paup.backends.live import LivePlanner
from paup.checkpoints import load_checkpoint

torch.set_num_threads(2)


def test_checkpoint_training_and_resume(tmp_path):
    paths=[]
    rng=np.random.RandomState(5)
    for i in range(4):
        path=tmp_path/f'window{i}.npz'
        np.savez(path,features=rng.normal(size=(5,4,16,16)).astype('float32'),
                 occupancy=rng.normal(size=(5,5,16,16)).astype('float32'),target=np.zeros(2,'float32'),
                 future_waypoints=np.ones((10,2),'float32'))
        paths.append(path.name)
    manifests=[]
    for name,files in [('train',paths[:2]),('validation',paths[2:])]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(dict(schema='paup_feature_windows_v1',
            occupancy_source='observations',pose_aligned=True,windows=files)));manifests.append(str(p))
    config=dict(stage='planner',backend=dict(factory='paup.backends.cached:create_task',train_manifest=manifests[0],
        validation_manifest=manifests[1],feature_channels=4),training=dict(epochs=1,batch_size=2,workers=0,
        optimizer='AdamW',learning_rate=.0002,weight_decay=.0005,seed=42))
    file=tmp_path/'config.yaml';file.write_text(yaml.safe_dump(config));out=tmp_path/'run'
    main(['--config',str(file),'--device','cpu','--output',str(out)])
    checkpoint=load_checkpoint(out/'last.ckpt')
    assert checkpoint['epoch']==0 and np.isfinite(checkpoint['best_validation_loss'])
    assert len(checkpoint['optimizer']['state'])>0
    main(['--config',str(file),'--device','cpu','--output',str(out),'--resume',str(out/'last.ckpt')])


class ObservedFrontend(torch.nn.Module):
    """Synthetic contract fixture, not an experimental backend or example result."""
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(()));self.calls=0
    def encode(self,observed):
        self.calls+=1
        probabilities=torch.zeros(2,1,16,16);probabilities[:,:,8,8]=.9
        return dict(predicted=probabilities,anchors=probabilities,aoi_ms=[0,200],
            history_available=[False,True],horizon_valid=[True,True],
            request_sampling_affines=torch.tensor([[[1.,0,0],[0,1.,0]],[[1.,0,0],[0,1.,0]]]),
            observed=observed)
    def fuse(self,encoded,mask,frame):
        return encoded['observed']*self.scale
    def align_history(self,records):
        records=[records[0]]*(5-len(records))+records
        return dict(features=torch.stack(records)[None],occupancy=torch.ones(1,5,5,16,16),target=torch.zeros(1,2))


def test_live_causal_rollout_frozen_frontend_and_planner_gradients():
    frontend=ObservedFrontend()
    model=LivePlanner(frontend,dict(uncertainty=dict(mode='fixed',fixed_sigma_px=2,weight=1)),feature_channels=4)
    window=[dict(time_ms=200*i,pose=dict(gps_y=0,gps_x=0,theta=0),
                 observation=torch.ones(4,16,16)*i) for i in range(3)]
    model.train();result=model(window);assert result.shape==(1,10,2)
    result.square().mean().backward()
    assert frontend.scale.grad is None and not frontend.training
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.planner.parameters())
    assert frontend.calls==3
