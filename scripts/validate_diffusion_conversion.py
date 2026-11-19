#!/usr/bin/env python3
"""Diagnostic only: fresh AR weights -> fixed data -> denoising and generation.

No Factory Trainer, no checkpoint-6000, no automatic full training. Reports the
actual optimizer precision; BF16 Adam is an experimental setting, not an official
conversion recipe. Run --tiny --steps 2 --eval-every 2 --sample-steps 2 locally.
"""
import argparse
import json
import random
from contextlib import contextmanager
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llamafactory.train.diffusion.loss import (
    _get_conditioner, _token_embedding, denoise_logits, diffusion_batch_step, prepare_conversion_training,
)
from llamafactory.train.diffusion.rl import sample_response


def rows():
    train, valid = [], []
    for split, dest, offset, count in [('train', train, 0, 8), ('valid', valid, 8, 4)]:
        for i in range(offset, offset + count):
            a, b = i + 2, i + 3
            dest.append(dict(kind='math', prompt=f'只回答整数：{a}+{b}等于多少？', response=str(a+b)))
            dest.append(dict(kind='copy', prompt=f'原样输出：编号{i}的猫在睡觉。', response=f'编号{i}的猫在睡觉。'))
            name = f'add_{i}'
            dest.append(dict(kind='code', prompt=f'只输出Python代码，定义函数{name}(a,b)，返回两数之和。',
                             response=f'def {name}(a, b):\n    return a + b', name=name))
            dest.append(dict(kind='long', prompt=f'按顺序输出编号{i}的记录，每项一行，共40项。',
                             response='\n'.join(f'记录{i}第{j}项：检查完成。' for j in range(1,41))))
    return train, valid


class ByteTokenizer:
    bos_token = '<bos>'
    pad_token_id = 0
    def encode(self, s, **kwargs): return [x+3 for x in s.encode()]
    def decode(self, ids, **kwargs): return bytes(x-3 for x in ids if 3 <= x < 259).decode(errors='replace')


def encode(tok, row, device, cap):
    text = tok.bos_token + '<|turn>user\n' + row['prompt'] + '<turn|>\n<|turn>model\n'
    p = torch.tensor(tok.encode(text, add_special_tokens=False), device=device)
    r = torch.tensor(tok.encode(row['response']+'<turn|>', add_special_tokens=False)[:cap], device=device)
    return p, r


def check_answer(row, text):
    if row['kind'] == 'code':
        code = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
        code = code.group(1) if code else text.split('<turn|>')[0].split('<eos>')[0]
        # Generated code runs in an isolated child, with timeout. Diagnostic tasks
        # are synthetic additions; this is not a sandbox for untrusted users.
        program = code + f"\nassert {row['name']}(2,3)==5\nassert {row['name']}(-2,1)==-1\n"
        try:
            result = subprocess.run([sys.executable, '-I', '-c', program], timeout=3,
                                    capture_output=True, text=True)
            return dict(pass_test=result.returncode == 0, error=result.stderr[-500:])
        except subprocess.TimeoutExpired:
            return dict(pass_test=False, error='timeout')
    clean = text.split('<turn|>')[0].split('<eos>')[0].strip()
    return dict(exact=clean == row['response'], contains_target=row['response'] in text)


@contextmanager
def preserve_rng():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(py)


def evaluate(model, tok, splits, args, step, out):
    model.eval()
    emb = _token_embedding(model)
    cfg = model.config.text_config
    report = dict(step=step, recovery=[], samples=[])
    with torch.inference_mode():
        for split, data in splits:
            for i, row in enumerate(data):
                p, target = encode(tok, row, emb.weight.device, args.canvas * 18)
                # Every block is tested against its clean history, including block 2+.
                for begin in range(0, len(target), args.canvas):
                    body_len = min(args.canvas, len(target) - begin)
                    clean = target.new_full((args.canvas,), 1)
                    clean[:body_len] = target[begin:begin+args.canvas]
                    history = torch.cat([p, target[:begin]])
                    for fraction in (.1, .5, 1.):
                        torch.manual_seed(10000+i*100+begin+int(fraction*10))
                        changed = torch.rand(clean.shape, device=clean.device) < fraction
                        noise = torch.randint(0,cfg.vocab_size,clean.shape,device=clean.device)
                        noisy = torch.where(changed,noise,clean)
                        logits = denoise_logits(model,history,noisy,pad_id=0)[0]
                        pred = logits.argmax(-1)
                        ce = torch.nn.functional.cross_entropy(logits.float(),clean,reduction='none')
                        report['recovery'].append(dict(split=split,row=i,block=begin//args.canvas,
                            noise=fraction,changed=int(changed.sum()),
                            restored=int(((pred==clean)&changed).sum()),
                            kept=int((~changed).sum()), preserved=int(((pred==clean)&~changed).sum()),
                            body_ce_sum=float(ce[:body_len].sum()),
                            tail_ce_sum=float(ce[body_len:].sum()),
                            tail_tokens=args.canvas-body_len,
                            kind=row['kind'], body_tokens=body_len,
                            body_changed=int(changed[:body_len].sum()),
                            body_restored=int(((pred[:body_len]==clean[:body_len])&changed[:body_len]).sum()),
                            tail_changed=int(changed[body_len:].sum()),
                            tail_restored=int(((pred[body_len:]==clean[body_len:])&changed[body_len:]).sum())))
                if i < args.sample_rows:
                    for steps in args.sample_steps:
                        for seed in args.sample_seeds:
                            torch.manual_seed(seed)
                            ids=sample_response(model,p,steps=steps,canvas_size=args.canvas,
                                                max_canvases=args.sample_canvases,vocab_size=cfg.vocab_size)
                            text=tok.decode(ids.tolist(),skip_special_tokens=False)
                            item=dict(split=split,row=i,kind=row['kind'],steps=steps,seed=seed,text=text,
                                      expected=row['response'],checks=check_answer(row,text))
                            report['samples'].append(item)
                            print(json.dumps(item,ensure_ascii=False),flush=True)
    for split,_ in splits:
        for fraction in (.1,.5,1.):
            rs=[r for r in report['recovery'] if r['split']==split and r['noise']==fraction]
            n=sum(r['changed'] for r in rs); k=sum(r['kept'] for r in rs)
            print('RECOVERY',step,split,fraction,
                  'restored',sum(r['restored'] for r in rs)/max(n,1),
                  'preserved',sum(r['preserved'] for r in rs)/max(k,1),flush=True)
    (out/f'eval_{step:04d}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    model.train()


def save_diagnostic_checkpoint(model, tok, optimizer, out, step, order, args):
    """Write a complete diagnostic recovery point; retain only the newest one."""
    import shutil
    from safetensors.torch import save_file
    staging=out/f'.checkpoint-{step}.writing'
    final=out/f'checkpoint-{step}'
    staging.mkdir()
    model.save_pretrained(staging)
    if hasattr(tok, 'save_pretrained'):
        tok.save_pretrained(staging)
    save_file({k:v.detach().cpu().contiguous() for k,v in _get_conditioner(model).state_dict().items()},
              str(staging/'self_conditioner.safetensors'))
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,order=order,
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    python_rng=random.getstate(),settings=vars(args)),staging/'training_state.pt')
    (staging/'COMPLETE').write_text(str(step))
    staging.rename(final)
    for old in out.glob('checkpoint-*'):
        if old.is_dir() and old!=final and (old/'COMPLETE').exists():shutil.rmtree(old)
    print('CHECKPOINT_COMPLETE',final,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='google/gemma-4-E2B-it')
    ap.add_argument('--out',default='/content/drive/MyDrive/saves/e2b_conversion_diagnostic')
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--eval-every',type=int,default=100)
    ap.add_argument('--sample-steps',type=int,nargs='+',default=[48,192])
    ap.add_argument('--sample-rows',type=int,default=4)
    ap.add_argument('--sample-seeds',type=int,nargs='+',default=[500])
    ap.add_argument('--sample-canvases',type=int,default=3)
    ap.add_argument('--tiny',action='store_true')
    ap.add_argument('--pilot',action='store_true',help='Timing only: no evaluation or model saving; at most 10 steps')
    ap.add_argument('--optimizer-mode',choices=['bf16-adam','fp32-master'],default='bf16-adam')
    ap.add_argument('--loss-mode',choices=['canvas-mean','body-tail-balanced'],default='canvas-mean')
    ap.add_argument('--save-intermediate',action='store_true',help='Save weights at each evaluation; training state at end')
    args=ap.parse_args()
    if args.pilot:
        args.steps = min(args.steps, 10)
    random.seed(42);torch.manual_seed(42)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists(): raise SystemExit('Use a fresh output directory; refusing to overwrite diagnosis.')
    (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
    args.canvas=8 if args.tiny else 256
    if args.tiny:
        tok=ByteTokenizer()
        cfg=Gemma4TextConfig(vocab_size=259,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            hidden_size_per_layer_input=4,sliding_window=32,
            layer_types=['sliding_attention','full_attention'])
        model=Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg)).to(torch.bfloat16)
    else:
        if not torch.cuda.is_available(): raise SystemExit('GPU required for E2B')
        tok=AutoTokenizer.from_pretrained(args.model)
        model=Gemma4ForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16).cuda()
    prepare_conversion_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if args.optimizer_mode == 'fp32-master':
        from llamafactory.train.diffusion.precision import MasterAdamW, require_master_memory
        print('MASTER_MEMORY', require_master_memory(model), flush=True)
        optimizer=MasterAdamW(model.parameters())
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1.125e-4,betas=(.95,.99),weight_decay=1e-4)
    train,valid=rows()
    if args.tiny: train,valid=train[:2],valid[:2]
    for split,data in [('train',train),('valid',valid)]:
        (out/f'{split}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    splits=[('train',train),('valid',valid)]
    if not args.pilot:
        with preserve_rng():
            evaluate(model,tok,splits,args,0,out)
    encoded=[encode(tok,r,next(model.parameters()).device,args.canvas*18) for r in train]
    model.train()
    order=list(range(len(encoded)));random.shuffle(order)
    with (out/'train_log.jsonl').open('w') as log:
        for step in range(1,args.steps+1):
            started=time.perf_counter()
            row_idx=order[(step-1)%len(order)]
            p,r=encoded[row_idx]
            torch.manual_seed(420000 + step)
            optimizer.zero_grad(set_to_none=True)
            # Fixed 50-step warmup; diagnostic schedule is recorded, not claimed official.
            lr=1.125e-4*min(step/50,1.)
            for group in optimizer.param_groups: group['lr']=lr
            tracked=model.model.language_model.layers[0].mlp.up_proj.weight
            before=tracked.detach().clone()
            loss,parts=diffusion_batch_step(model,[p],[r],canvas_size=args.canvas,num_canvases=18,pad_id=0,eos_id=1,diagnostics=True,loss_mode=args.loss_mode)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            state=optimizer.state[tracked]
            record=dict(step=step,row=row_idx,kind=train[row_idx]['kind'],
                diagnostics=parts['rows'][0],loss=float(loss.detach()),grad=float(norm),lr=lr,
                seconds=time.perf_counter()-started,
                changed_fraction=float((tracked.detach()!=before).float().mean()),
                state_dtypes={k:str(v.dtype) for k,v in state.items() if torch.is_tensor(v)})
            log.write(json.dumps(record)+'\n');log.flush()
            if step==1 or step%10==0: print('TRAIN',record,flush=True)
            if not args.pilot and (step%args.eval_every==0 or step==args.steps):
                if args.save_intermediate:
                    save_diagnostic_checkpoint(model,tok,optimizer,out,step,order,args)
                with preserve_rng():
                    evaluate(model,tok,splits,args,step,out)
    if not args.tiny and not args.pilot:
        from safetensors.torch import save_file
        final = out / 'final_model'
        model.save_pretrained(final)
        tok.save_pretrained(final)
        save_file({k:v.detach().cpu().contiguous() for k,v in _get_conditioner(model).state_dict().items()},
                  str(final/'self_conditioner.safetensors'))
    print('DIAGNOSTIC_DONE',out,flush=True)

if __name__=='__main__': main()
