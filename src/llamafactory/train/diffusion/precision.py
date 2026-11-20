"""Diagnostic AdamW with FP32 CPU master weights and moments.

Forward/backward still use the model dtype. This isolates optimizer storage and
rounding from BF16 AdamW; it is not a full FP32-forward experiment or an official
Google optimizer. CPU offload trades throughput for memory.
"""
from copy import deepcopy
import math
import torch


class MasterAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1.125e-4, betas=(.95,.99), eps=1e-8,
                 weight_decay=1e-4, chunk_size=16777216):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.chunk_size=chunk_size

    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        saved_state = state_dict["state"]
        if len(self.param_groups) != len(saved_groups):
            raise ValueError("Loaded state dict has a different number of parameter groups")
        for group, saved in zip(self.param_groups, saved_groups):
            if len(group["params"]) != len(saved["params"]):
                raise ValueError("Loaded state dict contains a parameter group that doesn't match optimizer size")
        super().load_state_dict({"state": {}, "param_groups": deepcopy(saved_groups)})
        self.state.clear()
        for group, saved in zip(self.param_groups, saved_groups):
            for param, saved_id in zip(group["params"], saved["params"]):
                raw = saved_state.get(saved_id)
                if raw is None:
                    raw = saved_state.get(str(saved_id))
                if not raw:
                    continue
                restored = {}
                for key, val in raw.items():
                    if torch.is_tensor(val):
                        restored[key] = val.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
                    else:
                        restored[key] = deepcopy(val)
                self.state[param] = restored

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures unsupported in diagnostic optimizer')
        for group in self.param_groups:
            b1,b2=group['betas'];lr=group['lr'];wd=group['weight_decay']
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise ValueError('Sparse gradients unsupported')
                state=self.state[p]
                if not state:
                    state['master']=p.detach().to(device='cpu',dtype=torch.float32,copy=True).contiguous()
                    state['exp_avg']=torch.zeros_like(state['master'])
                    state['exp_avg_sq']=torch.zeros_like(state['master'])
                    state['step']=torch.tensor(0.,dtype=torch.float32)
                state['step'].add_(1)
                step=int(state['step']);bc1=1-b1**step;bc2=math.sqrt(1-b2**step)
                master=state['master'].view(-1);m=state['exp_avg'].view(-1);v=state['exp_avg_sq'].view(-1)
                if not p.is_contiguous():
                    raise ValueError('Diagnostic optimizer requires contiguous parameters')
                flat=p.view(-1);grad=p.grad.contiguous().view(-1)
                for start in range(0,p.numel(),self.chunk_size):
                    sl=slice(start,start+self.chunk_size)
                    g=grad[sl].to(device='cpu').to(dtype=torch.float32)
                    if wd:
                        master[sl].mul_(1-lr*wd)
                    m[sl].lerp_(g,1-b1)
                    v[sl].mul_(b2).addcmul_(g,g,value=1-b2)
                    denom=v[sl].sqrt().div_(bc2).add_(group['eps'])
                    master[sl].addcdiv_(m[sl],denom,value=-lr/bc1)
                    flat[sl].copy_(master[sl].to(dtype=flat.dtype))
        return None


def require_master_memory(model):
    import psutil
    # Three FP32 copies on CPU; gradients are transferred one chunk at a time.
    need=sum(p.numel()*12 for p in model.parameters() if p.requires_grad) + 12*1024**3
    free=psutil.virtual_memory().available
    if free<need:
        raise RuntimeError(f'FP32 CPU master needs about {need/1024**3:.1f} GiB free host RAM including reserve; '
                           f'available {free/1024**3:.1f} GiB. No training started.')
    return dict(cpu_required_gib=need/1024**3,cpu_available_gib=free/1024**3)


class GpuMasterAdamW(torch.optim.Optimizer):
    """On-device FP32 master-weight AdamW for full-parameter BF16 models (e.g., 2B-4B on 80-96GB GPU).

    Eliminates BF16 ULP underflow at low learning rates while avoiding CPU offload overhead.
    """

    def __init__(self, params, lr=1e-4, betas=(0.95, 0.99), eps=1e-8, weight_decay=1e-4):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        saved_state = state_dict["state"]
        if len(self.param_groups) != len(saved_groups):
            raise ValueError("Loaded state dict has a different number of parameter groups")
        for group, saved in zip(self.param_groups, saved_groups):
            if len(group["params"]) != len(saved["params"]):
                raise ValueError("Loaded state dict contains a parameter group that doesn't match optimizer size")
        super().load_state_dict({"state": {}, "param_groups": deepcopy(saved_groups)})
        self.state.clear()
        for group, saved in zip(self.param_groups, saved_groups):
            for param, saved_id in zip(group["params"], saved["params"]):
                raw = saved_state.get(saved_id)
                if raw is None:
                    raw = saved_state.get(str(saved_id))
                if not raw:
                    continue
                restored = {}
                for key, val in raw.items():
                    if torch.is_tensor(val):
                        target_dev = param.device if key != "step" else torch.device("cpu")
                        restored[key] = val.detach().to(device=target_dev, dtype=torch.float32, copy=True).contiguous()
                    else:
                        restored[key] = deepcopy(val)
                self.state[param] = restored

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Closures unsupported in GpuMasterAdamW")
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Sparse gradients unsupported")
                state = self.state[p]
                if not state:
                    state["master"] = p.detach().to(dtype=torch.float32, copy=True).contiguous()
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["step"].add_(1.0)
                step = int(state["step"].item())
                bc1 = 1.0 - b1**step
                bc2 = math.sqrt(1.0 - b2**step)
                master = state["master"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                g = p.grad.detach().to(dtype=torch.float32)
                if wd:
                    master.mul_(1.0 - lr * wd)
                m.lerp_(g, 1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                denom = v.sqrt().div_(bc2).add_(eps)
                master.addcdiv_(m, denom, value=-lr / bc1)
                p.copy_(master.to(dtype=p.dtype))
        return None
