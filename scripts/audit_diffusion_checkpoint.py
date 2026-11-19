#!/usr/bin/env python3
"""Read-only diagnostic checkpoint audit. No optimizer or training is created.

Recovery and generation are separate phases so GPU time can be budgeted.
Generation covers every saved row; short tasks use multiple fixed seeds.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llamafactory.train.diffusion.loss import (
    _get_conditioner, _token_embedding, _embed_distribution, denoise_logits, denoise_step_logits,
)
from llamafactory.train.diffusion.rl import _commit, _temperature
from validate_diffusion_conversion import encode, check_answer


@torch.inference_mode()
def traced_response(model, prompt, steps, cap, canvas_size=256, early_stop=True):
    cfg=model.config.text_config
    emb=_token_embedding(model)
    chunks=[]; traces=[]; history=prompt.view(-1)
    for block in range(cap):
        canvas=torch.randint(0,cfg.vocab_size,(1,canvas_size),device=prompt.device)
        sc=torch.zeros(1,canvas_size,cfg.hidden_size,device=prompt.device,dtype=emb.weight.dtype)
        reason='step_limit'
        for i in range(steps):
            logits=denoise_step_logits(model,history,canvas[0],sc_embeddings=sc,pad_id=0)
            nxt,shaped,entropy=_commit(logits,canvas,temperature=_temperature(i,steps),
                entropy_bound=.1,vocab_size=cfg.vocab_size,is_final_step=i+1==steps)
            stable=bool((shaped.argmax(-1)==canvas).all())
            confident=bool(entropy.mean()<=.005)
            sc=_embed_distribution(emb,shaped)
            if early_stop and stable and confident:
                canvas=nxt;reason='entropy_and_stability';break
            canvas=nxt
        row=canvas[0]
        hits=((row==1)|(row==106)).nonzero()
        stop=int(hits[0]) if hits.numel() else None
        traces.append(dict(block=block,used_steps=i+1,reason=reason,
            entropy=float(entropy.mean()),stable=stable,stop_index=stop))
        chunks.append(row if stop is None else row[:stop+1])
        if stop is not None: break
        history=torch.cat([history,row])
    return torch.cat(chunks),traces,stop is None


def recovery_canvas(target, begin, full_canvas=False, canvas_size=256, eos_id=1):
    """Match training's EOS-filled last canvas; keep body/tail metrics separate."""
    body = target[begin:begin + canvas_size]
    if full_canvas:
        clean = target.new_full((canvas_size,), eos_id)
        clean[:len(body)] = body
    else:
        clean = body
    body_mask = torch.arange(len(clean), device=clean.device) < len(body)
    return clean, body_mask


def summarize(path):
    data=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not data: return
    if data[0]['phase']=='recovery':
        groups=defaultdict(list)
        for r in data:
            for region, counts in r.get('regions', {'all':r}).items():
                if counts['changed'] + counts['kept']:
                    groups[(r['split'],r['kind'],r['noise'],r['block_group'],region)].append(counts)
        for key,rs in sorted(groups.items()):
            n=sum(r['changed'] for r in rs);k=sum(r['kept'] for r in rs)
            macro=[r['restored']/r['changed'] for r in rs if r['changed']]
            print('RECOVERY',key,dict(blocks=len(rs),changed_tokens=n,
                restored=round(sum(r['restored'] for r in rs)/n,4) if n else None,
                block_macro=round(sum(macro)/len(macro),4) if macro else None,
                preserved=round(sum(r['preserved'] for r in rs)/k,4) if k else None),flush=True)
    else:
        groups=defaultdict(list)
        for r in data:groups[(r['split'],r['kind'],r['steps'])].append(r)
        for key,rs in groups.items():
            print('GENERATION',key,dict(attempts=len(rs),passed=sum(r['passed'] for r in rs),
                capped=sum(r['hit_output_cap'] for r in rs),
                seconds=sum(r['seconds'] for r in rs)),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/content/drive/MyDrive/saves/e2b_conversion_diagnostic_v1')
    ap.add_argument('--phase',choices=['recovery','generation'],required=True)
    ap.add_argument('--steps',nargs='+',type=int,default=[48,192])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    ap.add_argument('--max-canvases',type=int,default=18)
    ap.add_argument('--full-canvas',action='store_true')
    ap.add_argument('--no-early-stop',action='store_true')
    ap.add_argument('--failed-only',action='store_true')
    ap.add_argument('--summary-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.root); ckpt=root/'final_model'
    suffix = '_full256' if args.phase=='recovery' and args.full_canvas else ''
    if args.phase=='generation' and args.no_early_stop: suffix += '_noearly'
    if args.failed_only: suffix += '_failed'
    output=root/f'audit_{args.phase}{suffix}.jsonl'
    if args.summary_only:
        summarize(output)
        if args.phase=='generation':
            for line in output.read_text().splitlines():
                r=json.loads(line)
                print('CASE',r['split'],r['row'],r['steps'],r['seed'],
                    'before',r.get('baseline_passed'),'after',r['passed'],
                    'used',[t['used_steps'] for t in r['traces']],
                    'text',repr(r['text'][:200]),flush=True)
        return
    def key(r):return (r['split'],r['row'],r.get('block'),r.get('noise'),r.get('steps'),r.get('seed'))
    baseline={}
    if args.failed_only:
        if args.phase!='generation':raise SystemExit('--failed-only requires generation')
        source=root/'audit_generation.jsonl'
        if not source.exists():raise SystemExit('Run the baseline generation audit first')
        baseline={key(r):r for r in (json.loads(l) for l in source.read_text().splitlines() if l.strip())}
    for n in ['config.json','model.safetensors','self_conditioner.safetensors']:
        if not (ckpt/n).exists(): raise SystemExit(f'Missing checkpoint file: {ckpt/n}')
    if not torch.cuda.is_available():raise SystemExit('Connect GPU first')
    tok=AutoTokenizer.from_pretrained(ckpt)
    model=Gemma4ForConditionalGeneration.from_pretrained(ckpt,dtype=torch.bfloat16).cuda().eval()
    cond=_get_conditioner(model)
    cond.load_state_dict(load_file(str(ckpt/'self_conditioner.safetensors')),strict=True)
    model.eval()
    existing=[json.loads(l) for l in output.read_text().splitlines()] if output.exists() else []
    done={key(r) for r in existing}
    with output.open('a') as f,torch.inference_mode():
        for split in ['train','valid']:
            rows=json.loads((root/f'{split}.json').read_text())
            for idx,row in enumerate(rows):
                p,target=encode(tok,row,'cuda',18*256)
                if args.phase=='recovery':
                    for begin in range(0,len(target),256):
                        clean,body_mask=recovery_canvas(target,begin,args.full_canvas)
                        history=torch.cat([p,target[:begin]])
                        for noise_level in [.1,.5,1.]:
                            r=dict(phase='recovery',split=split,row=idx,kind=row['kind'],
                                block=begin//256,block_group='first' if begin==0 else 'later',noise=noise_level)
                            if key(r) in done:continue
                            torch.manual_seed(10000+idx*100+begin+int(noise_level*10))
                            changed=torch.rand(clean.shape,device='cuda')<noise_level
                            noisy=torch.where(changed,torch.randint(0,model.config.text_config.vocab_size,
                                clean.shape,device='cuda'),clean)
                            pred=denoise_logits(model,history,noisy,pad_id=0).argmax(-1)[0]
                            r.update(changed=int(changed.sum()),restored=int(((pred==clean)&changed).sum()),
                                kept=int((~changed).sum()),preserved=int(((pred==clean)&~changed).sum()))
                            r['canvas_tokens']=len(clean)
                            r['regions']={}
                            for region,mask in [('body',body_mask),('eos_tail',~body_mask)]:
                                moved=changed & mask; untouched=(~changed) & mask
                                r['regions'][region]=dict(changed=int(moved.sum()),
                                    restored=int(((pred==clean)&moved).sum()),kept=int(untouched.sum()),
                                    preserved=int(((pred==clean)&untouched).sum()))
                            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                    print('RECOVERY_ROW',split,idx,row['kind'],'target_tokens',len(target),flush=True)
                else:
                    for steps in args.steps:
                        for seed in (args.seeds[:1] if row['kind']=='long' else args.seeds):
                            r=dict(phase='generation',split=split,row=idx,kind=row['kind'],steps=steps,seed=seed)
                            if key(r) in done:continue
                            if args.failed_only:
                                previous=baseline.get(key(r))
                                if previous is None or previous['passed'] or row['kind']=='long':continue
                                r['baseline_passed']=previous['passed']
                                r['baseline_text']=previous['text']
                                r['baseline_used_steps']=[t['used_steps'] for t in previous['traces']]
                            torch.manual_seed(seed);t=time.perf_counter()
                            # One extra canvas beyond target length, capped at K=18.
                            row_cap=min(args.max_canvases,(len(target)+255)//256+1)
                            ids,traces,capped=traced_response(model,p,steps,row_cap,early_stop=not args.no_early_stop)
                            text=tok.decode(ids.tolist(),skip_special_tokens=False)
                            checks=check_answer(row,text)
                            r.update(text=text,expected=row['response'],checks=checks,
                                passed=bool(checks.get('pass_test',checks.get('exact',False))),
                                target_tokens=len(target),output_tokens=len(ids),
                                hit_output_cap=capped,cap_tokens=256 * row_cap,
                                target_exceeds_cap=len(target)>row_cap*256,
                                traces=traces,seconds=time.perf_counter()-t)
                            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                            print('SAMPLE',split,idx,row['kind'],steps,seed,r['passed'],
                                  'steps_used',[x['used_steps'] for x in traces],
                                  'output_cap',capped,flush=True)
    summarize(output)
    print('AUDIT_DONE',output,flush=True)

if __name__=='__main__':main()
