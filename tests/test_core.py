import json
import numpy as np
import torch
from paup.aoi import AoIManager
from paup.channel import MeasuredV2XChannel
from paup.causal import causal_forecast,CALIBRATION_PROTOCOL
from paup.selection import bounded_masks,DEFAULTS
from paup.uncertainty import PlanningAwareAgeUncertainty
from paup.models.communication import scaled_agent_attention


def test_delivery_causality_loss_and_out_of_order():
    manager=AoIManager()
    manager.enqueue('sender',0,200,'old')
    manager.enqueue('sender',100,150,'new')
    manager.enqueue('sender',120,180,'lost',dropped=True)
    manager.advance(149)
    assert np.isinf(manager.age_ms('sender'))
    manager.advance(150);assert manager.age_ms('sender')==50
    manager.advance(250)
    assert manager.last_valid['sender'].message=='new'
    assert manager.age_ms('sender')==150


def test_fixed_channel_does_not_need_local_measurements():
    channel=MeasuredV2XChannel(mode='fixed',delay_ms=200,loss_rate=0)
    assert channel.sample(25,1000,10)['latency_ms']==200


def test_fractional_causal_forecast_uses_only_received_history():
    values=torch.tensor([.1,.6,.4]).reshape(3,1,1,1)
    meta=torch.tensor([[1.,0.],[1.,250.],[0.,450.]])
    inputs=[]
    def predict(older,latest,steps):
        inputs.append((float(older.item()),float(latest.item()),steps))
        return torch.stack([latest[:,0]+.1*(j+1) for j in range(steps)],1)
    output=causal_forecast(torch.arange(3).reshape(3,1),values,torch.tensor([3]),
        torch.zeros(1,3,3,2,3),meta,predict)
    assert np.allclose(inputs[0][:2],[.4,.6]) and inputs[0][2]==2
    assert torch.allclose(output[1][1],torch.tensor([[[.725]]]),atol=1e-6)
    assert output[0].flatten().tolist()==[0,1]


def test_base_preservation_budget_and_deterministic_ties():
    base=torch.zeros(2,1,3,3);base[1,0,0,0]=1
    candidate=torch.ones_like(base)
    quality=[{'gate':1.},{'gate':1.}]
    config=dict(DEFAULTS,extra_fraction=0,extra_min_cells=2,extra_max_cells=2)
    final,baseline,raw,rows=bounded_masks(base,candidate,quality,.5,config)
    assert final[1].flatten().nonzero().flatten().tolist()==[0,1,2]
    assert torch.all(final|~baseline)
    quality[1]['gate']=0.
    final,*_=bounded_masks(base,candidate,quality,.5,config)
    assert final[1].sum()==1


def test_planning_weight_and_fresh_identity(tmp_path):
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(protocol=CALIBRATION_PROTOCOL,source='simulation_dpp_gt_residuals',
        units='heatmap_pixels',aoi_ms=[0,800],sigma_px=[1,3],heatmap_hw=[15,15])))
    model=PlanningAwareAgeUncertainty(dict(mode='planning_aware',weight=1.,calibration_path=str(calibration)))
    values=torch.zeros(1,1,15,15);values[0,0,7,7]=1
    assert torch.equal(model(values,[250],torch.zeros_like(values)),values)
    assert torch.equal(model(values,[0],torch.ones_like(values)),values)
    expanded=model(values,[250],torch.ones_like(values))
    assert torch.all(expanded>=values) and expanded[0,0,7,7]==1
    assert expanded[0,0,7,8]>0


def test_agent_attention_single_agent_identity():
    data=torch.randn(1,4,3,5)
    torch.testing.assert_close(scaled_agent_attention(data),data[0])
