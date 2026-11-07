"""Call an independently installed cooperative-perception training backend.

No restricted backend source, architecture YAML or checkpoint is redistributed.
"""
from pathlib import Path
import sys,copy
import torch
from torch.utils.data import DataLoader
from paup.training import TrainingTask


def create_task(config):
    backend=config['backend'];training=config['training']
    for root in reversed(backend.get('package_roots',[])):
        root=Path(root).resolve()
        if not root.is_dir():raise FileNotFoundError('External backend package root is missing: '+str(root))
        sys.path.insert(0,str(root))
    try:
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools.train_utils import create_model,create_loss
    except ImportError as exc:
        raise ImportError('Install the external perception backend described in docs/backends.md') from exc
    hypes=load_yaml(str(Path(backend['architecture_config']).resolve()))
    hypes=copy.deepcopy(hypes)
    hypes['root_dir']=str(Path(backend['train_root']).resolve())
    hypes['validate_dir']=str(Path(backend['validation_root']).resolve())
    hypes.setdefault('train_params',{})['batch_size']=training['batch_size']
    train=build_dataset(hypes,visualize=False,train=True)
    validation=build_dataset(hypes,visualize=False,train=False)
    model=create_model(hypes);criterion=create_loss(hypes)
    single=bool(getattr(train,'supervise_single',False))
    def loss(model,batch):
        inputs=batch['ego'];inputs['epoch']=getattr(model,'training_epoch',0)
        outputs=model(inputs)
        value=criterion(outputs,inputs['label_dict'])
        if single and 'label_dict_single' in inputs:
            value=value+hypes['train_params'].get('single_weight',1.)*criterion(outputs,inputs['label_dict_single'],suffix='_single')
        return value
    args=dict(batch_size=training['batch_size'],num_workers=training.get('workers',4),pin_memory=torch.cuda.is_available())
    return TrainingTask(model,DataLoader(train,shuffle=True,collate_fn=train.collate_batch_train,**args),
        DataLoader(validation,shuffle=False,collate_fn=validation.collate_batch_train,**args),loss)
