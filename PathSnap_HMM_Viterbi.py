#!/usr/bin/env python3
"""PathSnap: standalone, model-agnostic HMM/Viterbi candidate decoder.

Run from the project root:
    python PathSnap_HMM_Viterbi.py --model mlp
    python PathSnap_HMM_Viterbi.py --model logistic

Requires model-produced student_test_candidate_probabilities.parquet and
preprocessing_outputs/student_test/*_candidates.parquet. Outputs are
observation-to-arc matches, NOT a fully expanded turn-by-turn road path.
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = {
    'mlp': ROOT / 'mlp_outputs' / 'student_test_candidate_probabilities.parquet',
    'logistic': ROOT / 'logistic_regression_outputs' / 'student_test_candidate_probabilities.parquet',
}


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


def main():
    parser = argparse.ArgumentParser(description='PathSnap HMM/Viterbi route decoder')
    parser.add_argument('--model', choices=MODELS, default='mlp')
    parser.add_argument('--predictions', type=Path, help='Override model prediction parquet path')
    parser.add_argument('--max-hops', type=int, default=3)
    parser.add_argument('--disconnect-penalty', type=float, default=8.0)
    parser.add_argument('--hop-penalty', type=float, default=0.7)
    parser.add_argument('--reset-gap', type=int, default=5,
                        help='Reset sequence when observation IDs jump by more than this')
    args = parser.parse_args()
    if args.max_hops < 1 or args.disconnect_penalty < 0 or args.hop_penalty < 0:
        parser.error('max-hops must be >=1 and penalties must be nonnegative')
    pred_path = args.predictions or MODELS[args.model]
    if not pred_path.exists():
        parser.error(f'Missing predictions: {pred_path}. Run the {args.model} model notebook first.')
    predictions = pd.read_parquet(pred_path)
    required = {'trip_id','observation_id','arc_id','probability'}
    if not required.issubset(predictions.columns):
        parser.error(f'Prediction file missing columns: {sorted(required-set(predictions.columns))}')
    predictions['trip_id'] = predictions.trip_id.astype(str).str.zfill(8)
    predictions['probability'] = pd.to_numeric(predictions.probability, errors='coerce')
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0,1).all():
        parser.error('Prediction probabilities must be finite values in [0,1]')
    if predictions.duplicated(['trip_id','observation_id','arc_id']).any():
        parser.error('Duplicate (trip_id, observation_id, arc_id) predictions found')

    out = ROOT / 'hmm_outputs' / args.model
    out.mkdir(parents=True, exist_ok=True)
    results, diagnostics = [], []
    for trip_id, scores in predictions.groupby('trip_id', sort=True):
        candidate_path = ROOT / 'preprocessing_outputs' / 'student_test' / f'{trip_id}_candidates.parquet'
        if not candidate_path.exists():
            raise FileNotFoundError(f'Missing candidate features: {candidate_path}')
        candidates = pd.read_parquet(candidate_path,
                                     columns=['observation_id','arc_id','from_node_id','to_node_id'])
        merged = scores.merge(candidates, on=['observation_id','arc_id'], how='left',
                              validate='one_to_one', indicator=True)
        if not merged['_merge'].eq('both').all():
            raise ValueError(f'Prediction/candidate mismatch for trip {trip_id}')
        if merged[['from_node_id','to_node_id']].isna().any().any():
            raise ValueError(f'Missing network endpoint IDs for trip {trip_id}')
        decoded, stats = decode_trip(merged, args.max_hops,
                                     args.disconnect_penalty, args.hop_penalty,
                                     args.reset_gap)
        decoded.insert(0, 'trip_id', trip_id)
        results.append(decoded)
        diagnostics.append({'trip_id':trip_id, **stats})
        print(f'{trip_id}: matched {stats["observations"]} observations, '
              f'{stats["disconnected_transitions"]} unsupported transitions, '
              f'{stats["resets"]} resets')
    pd.concat(results, ignore_index=True).to_csv(out / 'observation_matches.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'diagnostics.csv', index=False)
    print(f'\nSaved: {out / "observation_matches.csv"}')
    print(f'Saved: {out / "diagnostics.csv"}')
    print('NOTE: This is an approximate Viterbi decoder. Candidate-only topology may miss valid connections.')
    print('NOTE: Output is one selected arc per observation, not a fully expanded route.')

if __name__ == '__main__':
    main()
