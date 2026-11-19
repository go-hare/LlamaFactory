import pytest
import torch
import torch.nn.functional as F
from llamafactory.train.diffusion.loss import canvas_objective,diffusion_batch_step
from test_diffusion_ple import tiny_model


def test_balanced_loss_and_no_tail():
    logits=torch.tensor([[3.,0.],[0.,3.],[3.,0.],[0.,3.]],requires_grad=True)
    target=torch.tensor([1,1,0,1]);valid=torch.ones(4,dtype=torch.bool)
    a=canvas_objective(logits,target,valid,1,'canvas-mean')
    torch.testing.assert_close(a,F.cross_entropy(logits,target))
    b=canvas_objective(logits,target,valid,1,'body-tail-balanced')
    expected=.5*F.cross_entropy(logits[:1],target[:1])+.5*F.cross_entropy(logits[1:],target[1:])
    torch.testing.assert_close(b,expected)
    b.backward();assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(canvas_objective(logits,target,valid,4,'body-tail-balanced'),a)
    with pytest.raises(ValueError):canvas_objective(logits,target,valid,1,'bad')


def test_loss_modes_same_rng_metadata():
    m=tiny_model()
    p=[torch.tensor([2,3])];r=[torch.tensor([4,5])]
    reports=[]
    for mode in ['canvas-mean','body-tail-balanced']:
        torch.manual_seed(71)
        loss,parts=diffusion_batch_step(m,p,r,canvas_size=8,num_canvases=18,pad_id=0,eos_id=1,
                                      diagnostics=True,loss_mode=mode)
        assert torch.isfinite(loss)
        reports.append(parts['rows'][0])
    assert reports[0]==reports[1]
