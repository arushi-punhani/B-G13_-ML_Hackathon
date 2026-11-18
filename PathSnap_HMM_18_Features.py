#!/usr/bin/env python3
"""Standalone, untrained, 18-feature HMM baseline for PathSnap.

Uses 11 candidate features + 7 observation features to calculate
hand-designed (not learned) emission/transition scores.
No MLP, LR, or model artifact needed.

Usage:
 python PathSnap_HMM_18_Features.py
 python PathSnap_HMM_18_Features.py --partition student_test

Training-partition evaluation uses route-derived WEAK pseudo-labels.
Requires numpy, pandas, pyarrow.
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CAND = ['candidate_rank', 'distance_to_arc_m', 'arc_length_m',
        'heading_difference_deg', 'projection_fraction',
        'distance_to_start_m', 'distance_to_end_m',
        'from_in_degree', 'from_out_degree', 'to_in_degree', 'to_out_degree']
OBS = ['dt_s', 'step_distance_m', 'speed_mps', 'accel_ms2',
       'turn_abs_deg', 'candidate_count', 'large_gap']
FEATURES = CAND + OBS  # Exactly 18 actual inputs (node IDs used separately for topology).


def num(df, name, default=0.0):
    return pd.to_numeric(df[name], errors='coerce').fillna(default).to_numpy(dtype=float)


def emission_scores(df):
    """All 18 features affect scoring. Weights are heuristics, not fitted parameters."""
    d = np.maximum(num(df,'distance_to_arc_m'), 0)
    rank = np.maximum(num(df,'candidate_rank',1),1)
    length = np.maximum(num(df,'arc_length_m',1),1)
    heading = np.abs((num(df,'heading_difference_deg',0)+180)%360-180)
    frac = np.clip(num(df,'projection_fraction',0.5),0,1)
    ds = np.maximum(num(df,'distance_to_start_m'),0)
    de = np.maximum(num(df,'distance_to_end_m'),0)
    fi = np.maximum(num(df,'from_in_degree'),0)
    fo = np.maximum(num(df,'from_out_degree'),0)
    ti = np.maximum(num(df,'to_in_degree'),0)
    to = np.maximum(num(df,'to_out_degree'),0)
    dt = np.maximum(num(df,'dt_s'),0)
    step = np.maximum(num(df,'step_distance_m'),0)
    speed = np.maximum(num(df,'speed_mps'),0)
    accel = np.abs(num(df,'accel_ms2'))
    turn = np.clip(np.abs(num(df,'turn_abs_deg')),0,180)
    count = np.maximum(num(df,'candidate_count',1),1)
    gap = np.clip(num(df,'large_gap'),0,1)
    # 18 explicitly used, including observation-context variables.
    distance_sigma = 12 + 0.12*np.minimum(step,120) + 0.12*np.minimum(dt,60) + 7*gap
    heading_weight = np.clip(speed/4,0.05,1) * (1-0.55*gap) * (1-0.40*turn/180)
    logp = -(d/distance_sigma)**2/2
    logp -= 0.48*heading_weight*(heading/55)**2
    logp -= 0.07*np.log1p(rank-1)
    logp -= 0.055*np.log1p(length/70)
    logp -= 0.075*np.abs(frac-0.5)
    logp -= 0.06*np.log1p(ds/70) + 0.06*np.log1p(de/70)
    logp += 0.025*np.log1p(fi) + 0.030*np.log1p(fo)
    logp += 0.025*np.log1p(ti) + 0.030*np.log1p(to)
    logp -= 0.020*np.log1p(accel)*(d/25)
    logp -= 0.035*np.log1p(count)*(rank-1)
    # Normalize per observation for well-scaled log-emissions.
    s = pd.Series(logp,index=df.index)
    m = s.groupby(df.observation_id).transform('max').to_numpy()
    w = np.exp(logp-m)
    z = pd.Series(w,index=df.index).groupby(df.observation_id).transform('sum').to_numpy()
    return np.log(np.maximum(w/z,1e-12))


def hop_dist(graph, source, target, max_hops=3):
    if source == target: return 0
    frontier = {source}
    seen = {source}
    for hop in range(1,max_hops+1):
        nxt = set()
        for u in frontier:
            for v in graph.get(u,()):
                if v == target: return hop
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier: break
    return None


def decode(df, log_emissions):
    df = df.copy()
    df['_log_emit'] = log_emissions
    df = df.sort_values(['observation_id','candidate_rank','arc_id'])
    graph = defaultdict(set)
    for r in df[['from_node_id','to_node_id']].drop_duplicates().itertuples(index=False):
        graph[int(r.from_node_id)].add(int(r.to_node_id))
    groups = [g.reset_index(drop=True) for _,g in df.groupby('observation_id',sort=True)]
    if not groups: return pd.DataFrame(),{}
    scores = [groups[0]['_log_emit'].to_numpy()]
    backs = [np.full(len(groups[0]),-1,dtype=int)]
    disconnected = 0
    for i in range(1,len(groups)):
        prev,cur = groups[i-1],groups[i]
        new = np.full(len(cur),-np.inf)
        back = np.full(len(cur),-1,dtype=int)
        for j,c in enumerate(cur.itertuples(index=False)):
            best=-np.inf
            bestk=-1
            for k,a in enumerate(prev.itertuples(index=False)):
                if a.arc_id == c.arc_id:
                    penalty=0.0
                else:
                    hops=hop_dist(graph,int(a.to_node_id),int(c.from_node_id))
                    penalty=0.6*(1+hops) if hops is not None else 8.0
                val=scores[-1][k]-penalty
                if val>best: best=val;bestk=k
            new[j]=best+float(cur.iloc[j]['_log_emit'])
            back[j]=bestk
        if np.max(new)<-1e8:  # numerical safety for long trajectories
            new-=np.max(new)
        scores.append(new)
        backs.append(back)
    idx=int(np.argmax(scores[-1]))
    selected=[]
    for i in range(len(groups)-1,-1,-1):
        r=groups[i].iloc[idx]
        selected.append({'observation_id':int(r.observation_id),'arc_id':int(r.arc_id),
                         'emission_probability':float(np.exp(r['_log_emit']))})
        if i: idx=int(backs[i][idx])
    selected.reverse()
    # Diagnose chosen transitions against candidate-only graph.
    lookup={(int(r.observation_id),int(r.arc_id)):(int(r.from_node_id),int(r.to_node_id))
            for r in df.itertuples(index=False)}
    for a,b in zip(selected,selected[1:]):
        u,v=lookup[(a['observation_id'],a['arc_id'])],lookup[(b['observation_id'],b['arc_id'])]
        if a['arc_id']!=b['arc_id'] and hop_dist(graph,u[1],v[0]) is None:
            disconnected+=1
    return pd.DataFrame(selected),{'observations':len(groups),'disconnected_transitions':disconnected}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['train','student_test'],default='train')
    args=parser.parse_args()
    folder=ROOT/'preprocessing_outputs'/args.partition
    paths=sorted(folder.glob('*_candidates.parquet'))
    if not paths: parser.error(f'No candidate files in {folder}')
    labels=None
    if args.partition=='train':
        lp=ROOT/'candidate_labels_train.csv'
        if not lp.exists(): parser.error(f'Missing {lp}')
        labels=pd.read_csv(lp,dtype={'trip_id':str})
        labels['trip_id']=labels.trip_id.str.zfill(8)
        if labels.duplicated(['trip_id','observation_id','arc_id']).any():
            parser.error('Duplicate pseudo-label keys')
    out=ROOT/'hmm_outputs'/f'standalone_18_features_{args.partition}'
    out.mkdir(parents=True,exist_ok=True)
    all_matches=[]; all_diagnostics=[]; eval_rows=[]
    for i,path in enumerate(paths,1):
        tid=path.name.removesuffix('_candidates.parquet')
        c=pd.read_parquet(path)
        o=folder/f'{tid}_observations.parquet'
        if not o.exists(): raise FileNotFoundError(o)
        obs=pd.read_parquet(o)
        missing=set(CAND)-set(c.columns) | (set(OBS)-set(obs.columns))
        if missing: raise ValueError(f'{tid}: missing features {sorted(missing)}')
        if not {'from_node_id','to_node_id','arc_id','observation_id'}.issubset(c):
            raise ValueError(f'{tid}: missing graph identifiers')
        if c.empty: continue
        if c.duplicated(['observation_id','arc_id']).any(): raise ValueError(f'{tid}: duplicate candidates')
        obs=obs[['observation_id']+OBS]
        if obs.observation_id.duplicated().any(): raise ValueError(f'{tid}: duplicate observations')
        c=c.merge(obs,on='observation_id',how='left',validate='many_to_one')
        if c[OBS].isna().all(axis=1).any(): raise ValueError(f'{tid}: missing observation rows')
        c=c.sort_values(['observation_id','candidate_rank','arc_id']).reset_index(drop=True)
        logp=emission_scores(c)
        matched,stats=decode(c,logp)
        matched.insert(0,'trip_id',tid)
        all_matches.append(matched)
        all_diagnostics.append({'trip_id':tid,**stats})
        if labels is not None:
            truth=labels.loc[labels.trip_id==tid,['observation_id','arc_id','label']]
            if not truth.empty:
                joined=c[['observation_id','arc_id']].merge(truth,on=['observation_id','arc_id'],how='inner')
                counts=c.groupby('observation_id').size()
                full=joined.groupby('observation_id').size()
                valid=full.index[full.eq(counts.reindex(full.index))]
                joined=joined[joined.observation_id.isin(valid)]
                positives=joined.groupby('observation_id').label.sum()
                valid=positives.index[positives.eq(1)]
                joined=joined[joined.observation_id.isin(valid)].copy()
                pred=matched[['observation_id','arc_id']].assign(pred=1)
                joined=joined.merge(pred,on=['observation_id','arc_id'],how='left')
                joined['pred']=joined.pred.fillna(0).astype(int)
                if not joined.empty:
                    joined.insert(0,'trip_id',tid)
                    eval_rows.append(joined)
        print(f'[{i}/{len(paths)}] {tid}: {stats["observations"]} matched, {stats["disconnected_transitions"]} candidate-graph gaps',flush=True)
    if not all_matches: parser.error('No predictions produced')
    pd.concat(all_matches,ignore_index=True).to_csv(out/'observation_matches.csv',index=False)
    pd.DataFrame(all_diagnostics).to_csv(out/'diagnostics.csv',index=False)
    (out/'feature_list.json').write_text(json.dumps({'candidate_features':CAND,'observation_features':OBS,'total':len(FEATURES),
        'note':'Heuristic 18-feature HMM; not learned parameters; node IDs used for graph transitions'},indent=2))
    if eval_rows:
        scored=pd.concat(eval_rows,ignore_index=True)
        y=scored.label.to_numpy(dtype=int); pred=scored.pred.to_numpy(dtype=int)
        tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum())
        fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        metrics={'feature_count':18,'evaluated_observations':int(scored.groupby(['trip_id','observation_id']).ngroups),
                 'candidate_accuracy':(tp+tn)/len(y),'precision':precision,'recall':recall,'f1':f1,
                 'top1_pseudo_label_agreement':tp/scored.groupby(['trip_id','observation_id']).ngroups,
                 'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                 'reference':'weak route-derived pseudo-labels; not independent ground truth'}
        (out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
        print('\n=== 18-FEATURE STANDALONE HMM (PSEUDO-LABEL EVALUATION) ===')
        for key in ('candidate_accuracy','precision','recall','f1','top1_pseudo_label_agreement'):
            print(f'{key}: {metrics[key]:.2%}')
    print(f'\nSaved outputs: {out}')
    print('NOTE: 18 heuristic inputs, no classifier training. Results not directly comparable without same trip split.')


if __name__=='__main__':
    main()
