#!/usr/bin/env python3
"""Sequential diagnostic A/B, live compact output, complete raw logs on disk."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default='/content/drive/MyDrive/saves/e2b_loss_controls_v3')
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--tiny',action='store_true')
    args=ap.parse_args()
    root=Path(args.base);root.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('validate_diffusion_conversion.py')
    for mode in ['canvas-mean','body-tail-balanced']:
        out=root/mode
        if (out/'settings.json').exists():
            raise SystemExit(f'{out} already exists; use a fresh --base. No overwrite or implicit restart.')
        cmd=[sys.executable,'-u',str(script),'--out',str(out),'--steps',str(args.steps),
             '--eval-every','100','--loss-mode',mode,'--optimizer-mode','bf16-adam',
             '--sample-rows','32','--sample-seeds','42','43','--sample-steps','48',
             '--sample-canvases','3','--save-intermediate']
        if args.tiny:cmd+=['--tiny','--eval-every',str(args.steps),'--sample-steps','2']
        print('START',mode,'B is a non-official diagnostic objective',flush=True)
        with (root/f'{mode}.log').open('w') as log:
            with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
                for line in child.stdout:
                    log.write(line);log.flush()
                    if line.startswith('{"split"'):
                        item=json.loads(line)
                        print('SAMPLE',mode,item['split'],item['row'],item['kind'],
                              item['seed'],item['checks'],flush=True)
                    elif not ('Loading weights:' in line or 'Writing model shards:' in line):
                        print(line,end='',flush=True)
                code=child.wait()
                if code:raise SystemExit(f'{mode} failed with {code}; next arm not started. See {root}/{mode}.log')
    print('LOSS_CONTROLS_DONE',root,flush=True)

if __name__=='__main__':main()
