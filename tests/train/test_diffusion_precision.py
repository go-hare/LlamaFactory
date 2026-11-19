"""Precision-control regression checks (no pretrained weights)."""
import torch
from llamafactory.train.diffusion.precision import MasterAdamW
from llamafactory.train.diffusion.loss import diffusion_batch_step
from test_diffusion_ple import tiny_model


def test_master_matches_fp32_adamw():
    p=torch.nn.Parameter(torch.linspace(-1,1,31))
    ref=torch.nn.Parameter(p.detach().clone())
    opt=MasterAdamW([p],chunk_size=7)
    expected=torch.optim.AdamW([ref],lr=1.125e-4,betas=(.95,.99),weight_decay=1e-4)
    for step in range(10):
        g=torch.sin(torch.arange(31)+step)
        p.grad=g.clone();ref.grad=g.clone()
        opt.step();expected.step()
        torch.testing.assert_close(p,ref,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(opt.state[p]['master'],ref,rtol=1e-6,atol=1e-7)


def test_master_retains_sub_bf16_updates():
    p=torch.nn.Parameter(torch.ones(4,dtype=torch.bfloat16))
    opt=MasterAdamW([p],lr=.0001125)
    for _ in range(30):
        p.grad=torch.ones_like(p);opt.step()
    assert torch.all(p<1)
    assert opt.state[p]['master'].dtype==torch.float32
    assert opt.state[p]['exp_avg'].dtype==torch.float32


def test_master_state_dict_preserves_fp32_precision_on_bf16_params():
    p_decay = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    p_nodecay = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    groups = [
        {"params": [p_decay], "weight_decay": 1e-4},
        {"params": [p_nodecay], "weight_decay": 0.0},
    ]
    opt1 = MasterAdamW(groups, lr=1.125e-4, chunk_size=3)
    for i in range(5):
        p_decay.grad = torch.full_like(p_decay, 0.25 * (i + 1))
        p_nodecay.grad = torch.full_like(p_nodecay, 0.5 * (i + 1))
        opt1.step()
    saved = opt1.state_dict()

    # Continue opt1 for 5 more steps vs restoring into a fresh MasterAdamW
    p_decay_2 = torch.nn.Parameter(p_decay.detach().clone())
    p_nodecay_2 = torch.nn.Parameter(p_nodecay.detach().clone())
    opt2 = MasterAdamW(
        [
            {"params": [p_decay_2], "weight_decay": 1e-4},
            {"params": [p_nodecay_2], "weight_decay": 0.0},
        ],
        lr=1.125e-4,
        chunk_size=3,
    )
    opt2.load_state_dict(saved)
    for param in (p_decay_2, p_nodecay_2):
        for key in ("master", "exp_avg", "exp_avg_sq", "step"):
            assert opt2.state[param][key].dtype == torch.float32
            assert opt2.state[param][key].device.type == "cpu"

    for i in range(5, 10):
        g1 = torch.full_like(p_decay, 0.25 * (i + 1))
        g2 = torch.full_like(p_nodecay, 0.5 * (i + 1))
        p_decay.grad, p_nodecay.grad = g1.clone(), g2.clone()
        p_decay_2.grad, p_nodecay_2.grad = g1.clone(), g2.clone()
        opt1.step()
        opt2.step()

    torch.testing.assert_close(p_decay_2, p_decay, rtol=0, atol=0)
    torch.testing.assert_close(p_nodecay_2, p_nodecay, rtol=0, atol=0)
    torch.testing.assert_close(opt2.state[p_decay_2]["master"], opt1.state[p_decay]["master"], rtol=0, atol=0)


def test_diagnostics_do_not_change_training_loss():
    m=tiny_model()
    kw=dict(canvas_size=8,num_canvases=2,pad_id=0,eos_id=1)
    p=[torch.tensor([2,3])];r=[torch.tensor([4,5,6])]
    torch.manual_seed(42);a,_=diffusion_batch_step(m,p,r,**kw)
    torch.manual_seed(42);b,parts=diffusion_batch_step(m,p,r,diagnostics=True,**kw)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    row=parts['rows'][0]
    assert row['body_count']==3 and row['tail_count']==5
    assert 0<row['time']<1
    assert row['body_ce_sum']>=0 and row['tail_ce_sum']>=0
