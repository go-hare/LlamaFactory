#!/usr/bin/env python3
"""Print existing control reports; never loads models or uses a GPU."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default='/content/drive/MyDrive/saves/e2b_precision_controls')
    args=ap.parse_args()
    for mode in ['bf16-adam','fp32-master']:
        root=Path(args.base)/mode
        log=root/'train_log.jsonl'
        print('\nMODE',mode)
        if not log.exists():print('not run');continue
        groups=defaultdict(lambda:[0.,0,0.,0])
        records=[json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        for r in records:
            d=r['diagnostics'];bucket='high' if d['time']>=.8 else ('low' if d['time']<.2 else 'mid')
            g=groups[(r['kind'],bucket)]
            g[0]+=d['body_ce_sum'];g[1]+=d['body_count'];g[2]+=d['tail_ce_sum'];g[3]+=d['tail_count']
        print('completed steps',records[-1]['step'],'state dtypes',records[-1]['state_dtypes'])
        print('tracked matrix changed fraction first/last',records[0]['changed_fraction'],records[-1]['changed_fraction'])
        for k,g in sorted(groups.items()):
            print('TRAIN_CE',k,'body',round(g[0]/max(1,g[1]),4),'tail',round(g[2]/max(1,g[3]),4),
                  'body/tail tokens',g[1],g[3])
        for p in sorted(root.glob('eval_*.json')):
            report=json.loads(p.read_text());gg=defaultdict(list)
            for r in report['recovery']:gg[(r['split'],r['kind'],r['noise'])].append(r)
            for k,rs in sorted(gg.items()):
                n=sum(r['body_changed'] for r in rs)
                bc=sum(r['body_tokens'] for r in rs);tc=sum(r['tail_tokens'] for r in rs)
                print('EVAL',report['step'],k,'body_recovery',round(sum(r['body_restored'] for r in rs)/n,4) if n else None,
                    'body_CE',round(sum(r['body_ce_sum'] for r in rs)/max(1,bc),4),
                    'tail_CE',round(sum(r['tail_ce_sum'] for r in rs)/max(1,tc),4))
            for sample in report['samples']:
                print('SAMPLE',report['step'],sample['split'],sample['kind'],sample['steps'],sample['checks'],
                      repr(sample['text'][:150]))

if __name__=='__main__':main()
