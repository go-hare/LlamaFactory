#!/usr/bin/env python3
"""Read loss-control reports without loading any model."""
import argparse,json
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='/content/drive/MyDrive/saves/e2b_loss_controls_v3')
    args=p.parse_args()
    for mode in ['canvas-mean','body-tail-balanced']:
        root=Path(args.base)/mode
        print('\nMODE',mode,'(B diagnostic, not official)' if mode=='body-tail-balanced' else '')
        for f in sorted(root.glob('eval_*.json')):
            report=json.loads(f.read_text());groups=defaultdict(list)
            for r in report['samples']:groups[(r['split'],r['kind'],r['steps'])].append(r)
            for key,rs in sorted(groups.items()):
                passes=sum(bool(r['checks'].get('pass_test',r['checks'].get('exact',False))) for r in rs)
                print('GEN',report['step'],key,'passed',passes,'/',len(rs))
            groups=defaultdict(list)
            for r in report['recovery']:
                if r['noise']==1.:groups[(r['split'],r['kind'])].append(r)
            for key,rs in sorted(groups.items()):
                n=sum(r['body_changed'] for r in rs);b=sum(r['body_tokens'] for r in rs);t=sum(r['tail_tokens'] for r in rs)
                print('FULL_NOISE',report['step'],key,
                      'body_recovery',round(sum(r['body_restored'] for r in rs)/max(n,1),4),
                      'body_CE',round(sum(r['body_ce_sum'] for r in rs)/max(b,1),4),
                      'tail_CE',round(sum(r['tail_ce_sum'] for r in rs)/max(t,1),4))
        checkpoints=sorted(root.glob('checkpoint-*/COMPLETE'))
        print('complete checkpoints',[str(x.parent) for x in checkpoints])

if __name__=='__main__':main()
