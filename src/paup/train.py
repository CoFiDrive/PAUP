"""Train perception, a planner, or a planner fine-tune through an explicit backend."""
import argparse,importlib,json,os,random
from pathlib import Path
import numpy as np
import torch
import yaml
from paup.training import epoch
from paup.checkpoints import load_checkpoint


def resolve(spec):
    module,name=spec.split(':',1)
    return getattr(importlib.import_module(module),name)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',default=None)
    p.add_argument('--output',type=Path)
    p.add_argument('--initialize',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--check-config',action='store_true')
    a=p.parse_args(argv);cfg=yaml.safe_load(a.config.read_text())
    for key in ['backend','training']: 
        if key not in cfg:raise ValueError('Missing configuration section: '+key)
    t=cfg['training']
    if t['epochs']<=0 or t['batch_size']<=0 or t['learning_rate']<=0:
        raise ValueError('Epochs, batch size and learning rate must be positive')
    if a.initialize and a.resume:raise ValueError('Choose initialization or resume, not both')
    if a.check_config:
        print(json.dumps({'stage':cfg.get('stage'),'backend':cfg['backend']['factory'],
                          'configuration_valid':True,'training_started':False},indent=2));return
    seed=int(t.get('seed',42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg['_config_directory']=str(a.config.resolve().parent)
    task=resolve(cfg['backend']['factory'])(cfg)
    task.model.to(device)
    out=(a.output or Path(t.get('output','runs/'+cfg.get('stage','training')))).resolve()
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError('Use a fresh output directory or an explicit --resume checkpoint')
    out.mkdir(parents=True,exist_ok=True)
    if a.initialize:
        checkpoint=load_checkpoint(a.initialize)
        state=checkpoint.get('model_state_dict',checkpoint.get('model',checkpoint))
        loader=getattr(task.model,'load_compatible_state',None)
        loader(state) if loader else task.model.load_state_dict(state,strict=True)
    parameters=[p for p in task.model.parameters() if p.requires_grad]
    if not parameters:raise ValueError('No trainable parameters in the selected task')
    optimizers={'Adam':torch.optim.Adam,'AdamW':torch.optim.AdamW}
    optimizer=optimizers[t.get('optimizer','AdamW')](parameters,lr=t['learning_rate'],weight_decay=t.get('weight_decay',0.))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t['epochs'],eta_min=t.get('min_learning_rate',1e-6))
    start=0;best=float('inf')
    if a.resume:
        checkpoint=load_checkpoint(a.resume)
        task.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        start=checkpoint['epoch']+1;best=checkpoint['best_validation_loss']
        random.setstate(checkpoint['rng']['python']);np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if device.type=='cuda' and checkpoint['rng'].get('cuda'):torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    config_public={k:v for k,v in cfg.items() if not k.startswith('_')}
    (out/'config.yaml').write_text(yaml.safe_dump(config_public,sort_keys=False))
    for index in range(start,t['epochs']):
        task.model.training_epoch=index
        train_loss=epoch(task,task.train_loader,device,optimizer,t.get('clip_grad',10.))
        with torch.no_grad():validation_loss=epoch(task,task.validation_loader,device)
        improved=validation_loss<best;best=min(best,validation_loss)
        record={'epoch':index,'training_loss':train_loss,'validation_loss':validation_loss,
                'learning_rate':optimizer.param_groups[0]['lr']}
        print(json.dumps(record),flush=True)
        with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        scheduler.step()
        state={'epoch':index,'model_state_dict':task.model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'best_validation_loss':best,'configuration':config_public,
               'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                      'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}}
        tmp=out/'checkpoint.tmp';torch.save(state,tmp);os.replace(tmp,out/'last.ckpt')
        if improved:
            torch.save(state,out/'best.tmp');os.replace(out/'best.tmp',out/'best.ckpt')


if __name__=='__main__':main()
