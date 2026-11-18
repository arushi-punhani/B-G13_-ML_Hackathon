#!/usr/bin/env python3
"""Standalone geometry-only HMM/Viterbi with pseudo-label evaluation.

From project root:
    python PathSnap_Standalone_HMM_Evaluated.py
    python PathSnap_Standalone_HMM_Evaluated.py --partition student_test

Default train mode evaluates against candidate_labels_train.csv (weak labels).
No ML models, model outputs, or training are required.
Requires pandas, numpy, pyarrow and preprocessing_outputs/<partition>/*.parquet.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def directed_hops(start_arc, end_arc, endpoints, outgoing, max_hops=3):
    """Shortest number of directed arc transitions, up to max_hops.

    The candidate arc itself counts as hop zero. A transition to a successor
    arc is allowed when the first arc's to_node == successor's from_node.
    """
    if start_arc == end_arc:
        return 0
    if start_arc not in endpoints or end_arc not in endpoints:
        return None
    target_start = endpoints[end_arc][0]
    start_end = endpoints[start_arc][1]
    if start_end == target_start:
        return 1
    # For 2+ arc transitions, walk intermediate arcs by their ending nodes.
    frontier = {start_end}
    seen = set(frontier)
    for hops in range(2, max_hops + 1):
        following = set()
        for node in frontier:
            for next_node in outgoing.get(node, ()):
                if next_node == target_start:
                    return hops
                if next_node not in seen:
                    following.add(next_node)
        if not following:
            break
        seen.update(following)
        frontier = following
    return None

def decode_trip(df, max_hops, disconnect_penalty, hop_penalty, reset_gap):
    # DP state per GPS observation, up to 8 candidate arcs per observation.
    endpoints = dict(zip(df.arc_id.astype('int64'),
                         zip(df.from_node_id.astype('int64'), df.to_node_id.astype('int64'))))
    outgoing = defaultdict(set)
    for a, b in endpoints.values():
        outgoing[a].add(b)
    # Include full trip network connectivity if caller supplies extra network arcs
    # via --raw-arcs; otherwise only candidate subgraph is available.
    cache = {}
    observations = []
    for obs, g in df.groupby('observation_id', sort=True):
        arcs = g.arc_id.to_numpy(dtype=np.int64)
        probs = np.clip(g.probability.to_numpy(dtype=float), 1e-8, 1-1e-8)
        if not len(arcs):
            continue
        observations.append((int(obs), arcs, probs))
    if not observations:
        return pd.DataFrame(), {'observations':0,'disconnected_transitions':0,'resets':0}

    all_costs, backpointers, transitions = [], [], []
    resets = 0
    for t, (obs, arcs, probs) in enumerate(observations):
        emission = -np.log(probs)
        if t == 0 or obs - observations[t-1][0] > reset_gap:
            all_costs.append(emission)
            backpointers.append(np.full(len(arcs), -1, dtype=int))
            transitions.append(np.full(len(arcs), -1, dtype=int))
            if t:
                resets += 1
            continue
        prev_arcs = observations[t-1][1]
        costs = np.full(len(arcs), np.inf)
        back = np.full(len(arcs), -1, dtype=int)
        transition_types = np.full(len(arcs), -1, dtype=int)
        for j, arc in enumerate(arcs):
            for i, prev in enumerate(prev_arcs):
                pair = (int(prev), int(arc))
                if pair not in cache:
                    cache[pair] = directed_hops(pair[0], pair[1], endpoints, outgoing, max_hops)
                hops = cache[pair]
                penalty = disconnect_penalty if hops is None else hop_penalty * max(0, hops-1)
                score = all_costs[-1][i] + emission[j] + penalty
                if score < costs[j]:
                    costs[j] = score
                    back[j] = i
                    transition_types[j] = -2 if hops is None else hops
        all_costs.append(costs)
        backpointers.append(back)
        transitions.append(transition_types)

    # Backtrace each reset-delimited segment.
    selected = np.full(len(observations), -1, dtype=int)
    ends = [i for i in range(len(observations)) if i == len(observations)-1
            or np.all(backpointers[i+1] == -1)]
    for end in ends:
        j = int(np.argmin(all_costs[end]))
        i = end
        while i >= 0:
            selected[i] = j
            prior = int(backpointers[i][j])
            if prior < 0:
                break
            j = prior
            i -= 1
    rows = []
    disconnected = 0
    for t, (obs, arcs, probs) in enumerate(observations):
        j = selected[t]
        if j < 0:
            raise RuntimeError(f'Backtrace failed at observation {obs}')
        status = int(transitions[t][j])
        if status == -2:
            disconnected += 1
        rows.append({'observation_id':obs, 'arc_id':int(arcs[j]),
                     'probability':float(probs[j]),
                     'transition_hops':status,
                     'disconnected_transition':status == -2})
    return pd.DataFrame(rows), {'observations':len(rows),
                                'disconnected_transitions':disconnected,
                                'resets':resets}

def standalone_emissions(df, sigma_distance, sigma_heading, heading_weight):
    """Normalize geometry-only emission likelihoods within each observation.

    Distance uses a Gaussian likelihood; heading is an optional Gaussian
    likelihood on angular difference (when heading is available). This is a
    heuristic HMM emission model, not a trained or calibrated probability.
    """
    d = pd.to_numeric(df['distance_to_arc_m'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('distance_to_arc_m must be finite and nonnegative')
    log_score = -0.5 * (d / sigma_distance) ** 2
    if heading_weight > 0 and 'heading_difference_deg' in df.columns:
        angle = pd.to_numeric(df['heading_difference_deg'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(angle)
        angle = np.abs((angle + 180) % 360 - 180)
        log_score[valid] -= heading_weight * 0.5 * (angle[valid] / sigma_heading) ** 2
    # Stable softmax per observation. Clip only after normalization to ensure
    # the existing decoder can take logs safely.
    scores = pd.Series(log_score, index=df.index)
    maxima = scores.groupby(df['observation_id']).transform('max')
    weights = np.exp((scores - maxima).to_numpy())
    totals = pd.Series(weights, index=df.index).groupby(df['observation_id']).transform('sum').to_numpy()
    return np.clip(weights / totals, 1e-8, 1 - 1e-8)


def main():
    p = argparse.ArgumentParser(description='Standalone HMM + Viterbi + pseudo-label metrics')
    p.add_argument('--partition', choices=['train','student_test'], default='train')
    p.add_argument('--distance-sigma', type=float, default=20.0)
    p.add_argument('--heading-sigma', type=float, default=45.0)
    p.add_argument('--heading-weight', type=float, default=0.5)
    p.add_argument('--max-hops', type=int, default=3)
    p.add_argument('--disconnect-penalty', type=float, default=8.0)
    p.add_argument('--hop-penalty', type=float, default=0.7)
    p.add_argument('--reset-gap', type=int, default=5)
    args = p.parse_args()
    if args.distance_sigma <= 0 or args.heading_sigma <= 0 or args.heading_weight < 0:
        p.error('Sigmas must be positive and heading weight nonnegative')
    if args.max_hops < 1 or min(args.disconnect_penalty, args.hop_penalty) < 0 or args.reset_gap < 1:
        p.error('Invalid connectivity settings')

    folder = ROOT / 'preprocessing_outputs' / args.partition
    paths = sorted(folder.glob('*_candidates.parquet'))
    if not paths:
        p.error(f'No candidate parquet files found in {folder}')
    labels = None
    if args.partition == 'train':
        label_path = ROOT / 'candidate_labels_train.csv'
        if not label_path.exists():
            p.error(f'Labels missing: {label_path}')
        labels = pd.read_csv(label_path, dtype={'trip_id':str})
        needed = {'trip_id','observation_id','arc_id','label'}
        if not needed.issubset(labels.columns):
            p.error(f'Labels missing columns: {sorted(needed - set(labels.columns))}')
        labels['trip_id'] = labels['trip_id'].str.zfill(8)
        if labels.duplicated(['trip_id','observation_id','arc_id']).any():
            p.error('Duplicate pseudo-label keys')
        if not labels['label'].isin([0,1]).all():
            p.error('Pseudo-labels must be binary 0/1')

    out = ROOT / 'hmm_outputs' / ('standalone_train_evaluated' if labels is not None else 'standalone_student_test')
    out.mkdir(parents=True, exist_ok=True)
    all_matches, all_diagnostics, all_eval = [], [], []
    for k,path in enumerate(paths,1):
        trip_id = path.name.removesuffix('_candidates.parquet')
        df = pd.read_parquet(path)
        required = {'observation_id','arc_id','from_node_id','to_node_id','distance_to_arc_m'}
        if not required.issubset(df.columns):
            raise ValueError(f'{path.name}: missing {sorted(required-set(df.columns))}')
        if df.empty:
            print(f'[{k}/{len(paths)}] {trip_id}: no candidates; skipped', flush=True)
            continue
        if df[list(required)].isna().any().any():
            raise ValueError(f'{path.name}: missing required candidate values')
        if df.duplicated(['observation_id','arc_id']).any():
            raise ValueError(f'{path.name}: duplicate observation/arc candidates')
        df['probability'] = standalone_emissions(df,args.distance_sigma,args.heading_sigma,args.heading_weight)
        decoded,stats = decode_trip(df,args.max_hops,args.disconnect_penalty,args.hop_penalty,args.reset_gap)
        if decoded.empty:
            continue
        decoded.insert(0,'trip_id',trip_id)
        all_matches.append(decoded)
        all_diagnostics.append({'trip_id':trip_id,**stats})
        if labels is not None:
            trip_labels = labels.loc[labels.trip_id.eq(trip_id), ['observation_id','arc_id','label']]
            if not trip_labels.empty:
                # Inner join means only candidate rows with available pseudo-labels are evaluated.
                scored = df[['observation_id','arc_id']].merge(trip_labels,
                    on=['observation_id','arc_id'],how='inner',validate='one_to_one')
                selected = decoded[['observation_id','arc_id']].copy()
                selected['predicted_positive'] = 1
                scored = scored.merge(selected,on=['observation_id','arc_id'],how='left',validate='one_to_one')
                scored['predicted_positive'] = scored['predicted_positive'].fillna(0).astype(int)
                # Only score fully labeled observations; exclude partial candidate labels.
                candidate_counts = df.groupby('observation_id').size()
                scored_counts = scored.groupby('observation_id').size()
                complete_obs = scored_counts.index[scored_counts.eq(candidate_counts.reindex(scored_counts.index))]
                scored = scored.loc[scored.observation_id.isin(complete_obs)].copy()
                # Require exactly one pseudo-positive per observation to interpret top-1.
                positive_counts = scored.groupby('observation_id')['label'].sum()
                valid_obs = positive_counts.index[positive_counts.eq(1)]
                scored = scored.loc[scored.observation_id.isin(valid_obs)].copy()
                if not scored.empty:
                    scored.insert(0,'trip_id',trip_id)
                    all_eval.append(scored)
        print(f'[{k}/{len(paths)}] {trip_id}: {stats["observations"]} observations, '
              f'{stats["disconnected_transitions"]} disconnected transitions',flush=True)

    if not all_matches:
        p.error('No observations decoded')
    pd.concat(all_matches,ignore_index=True).to_csv(out/'observation_matches.csv',index=False)
    pd.DataFrame(all_diagnostics).to_csv(out/'diagnostics.csv',index=False)
    print(f'\nSaved matches and diagnostics to: {out}')
    if labels is not None:
        if not all_eval:
            print('No fully labeled observations with one positive candidate: metrics unavailable.')
            return
        scored = pd.concat(all_eval,ignore_index=True)
        y = scored.label.to_numpy(dtype=int)
        pred = scored.predicted_positive.to_numpy(dtype=int)
        tp = int(((y==1)&(pred==1)).sum())
        fp = int(((y==0)&(pred==1)).sum())
        tn = int(((y==0)&(pred==0)).sum())
        fn = int(((y==1)&(pred==0)).sum())
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        metrics = {
            'evaluation_reference':'route-derived pseudo-labels (NOT verified ground truth)',
            'partition':'train',
            'evaluated_trips':int(scored.trip_id.nunique()),
            'evaluated_observations':int(scored.groupby(['trip_id','observation_id']).ngroups),
            'evaluated_candidate_rows':int(len(scored)),
            'total_decoded_observations':int(sum(x['observations'] for x in all_diagnostics)),
            'candidate_accuracy':float((tp+tn)/len(scored)),
            'positive_precision':float(precision),
            'positive_recall':float(recall),
            'positive_f1':float(f1),
            'top1_pseudo_label_agreement':float(tp/scored.groupby(['trip_id','observation_id']).ngroups),
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'disconnected_transitions':int(sum(x['disconnected_transitions'] for x in all_diagnostics))
        }
        import json
        (out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('\n=== STANDALONE HMM vs PSEUDO-LABELS ===')
        print(f'Evaluated observations: {metrics["evaluated_observations"]:,} / {metrics["total_decoded_observations"]:,}')
        print(f'Candidate accuracy: {metrics["candidate_accuracy"]:.2%}')
        print(f'Positive precision: {metrics["positive_precision"]:.2%}')
        print(f'Positive recall: {metrics["positive_recall"]:.2%}')
        print(f'Positive F1: {metrics["positive_f1"]:.2%}')
        print(f'Top-1 agreement: {metrics["top1_pseudo_label_agreement"]:.2%}')
        print(f'Saved metrics: {out / "validation_metrics.json"}')
        print('CAUTION: These are pseudo-label agreement metrics, not true map-matching accuracy.')
    else:
        print('Student-test predictions saved; no accuracy/precision without reference labels.')
    print('Note: Connectivity uses candidate arcs, not full shortest-path expansion.')

if __name__ == '__main__':
    main()
