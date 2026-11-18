"""Evaluate a connectivity-aware Viterbi decoder on PathSnap MLP validation pseudo-labels.

Run: python PathSnap_HMM_Evaluation.py
Requires: pandas, numpy, pyarrow (for candidate feature parquet files).

This is a standalone validation decoder, not necessarily identical to the
student-test PathSnap_HMM_Viterbi.py decoder. Metrics use pseudo-labels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / 'mlp_outputs' / 'validation_candidates.csv'
FEATURES_DIR = ROOT / 'preprocessing_outputs' / 'train'
OUT = ROOT / 'hmm_outputs' / 'mlp_validation'
OUT.mkdir(parents=True, exist_ok=True)

# Tunable relative transition preferences. Larger negative values discourage a transition.
SAME_ARC_BONUS = 0.75
CONNECTED_BONUS = 0.25
DISCONNECTED_PENALTY = -4.0
EPS = 1e-12


def normalize_id(s):
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)


def load_features(trip_id):
    path = FEATURES_DIR / f'{trip_id}_candidates.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing road connectivity features: {path}')
    cols = ['observation_id', 'arc_id', 'from_node_id', 'to_node_id']
    df = pd.read_parquet(path, columns=cols)
    df['observation_id'] = pd.to_numeric(df['observation_id'], errors='raise').astype('int64')
    df['arc_id'] = pd.to_numeric(df['arc_id'], errors='raise').astype('int64')
    return df.drop_duplicates(['observation_id', 'arc_id'])


def decode_trip(df):
    df = df.sort_values(['observation_id', 'arc_id']).copy()
    groups = [g for _, g in df.groupby('observation_id', sort=True)]
    if not groups:
        return [], 0
    backpointers = []
    prev = None
    prev_scores = None
    for g in groups:
        p = np.clip(g['probability'].to_numpy(dtype=float), EPS, 1.0)
        emit = np.log(p)
        if prev is None:
            scores = emit
            backpointers.append(None)
        else:
            same = prev['arc_id'].to_numpy()[:, None] == g['arc_id'].to_numpy()[None, :]
            connected = prev['to_node_id'].to_numpy()[:, None] == g['from_node_id'].to_numpy()[None, :]
            # Do not count missing node IDs as connections.
            valid_nodes = pd.notna(prev['to_node_id'].to_numpy())[:, None] & pd.notna(g['from_node_id'].to_numpy())[None, :]
            connected &= valid_nodes
            transition = np.full(same.shape, DISCONNECTED_PENALTY)
            transition[connected] = CONNECTED_BONUS
            transition[same] = SAME_ARC_BONUS
            matrix = prev_scores[:, None] + transition
            back = matrix.argmax(axis=0)
            scores = emit + matrix[back, np.arange(len(g))]
            backpointers.append(back)
        prev, prev_scores = g, scores
    indices = [0] * len(groups)
    indices[-1] = int(np.argmax(prev_scores))
    for t in range(len(groups) - 1, 0, -1):
        indices[t-1] = int(backpointers[t][indices[t]])
    picks = [g.iloc[i] for g, i in zip(groups, indices)]
    disconnected = 0
    for a, b in zip(picks, picks[1:]):
        if a['arc_id'] != b['arc_id'] and (pd.isna(a['to_node_id']) or pd.isna(b['from_node_id']) or a['to_node_id'] != b['from_node_id']):
            disconnected += 1
    return picks, disconnected


def binary_metrics(y, predicted):
    y = np.asarray(y, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((y & predicted).sum())
    fp = int((~y & predicted).sum())
    fn = int((y & ~predicted).sum())
    tn = int((~y & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': (tp+tn)/len(y) if len(y) else 0.0,
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f'Missing {PREDICTIONS}; create it from MLP validation parquet first.')
    df = pd.read_csv(PREDICTIONS, dtype={'trip_id': str})
    required = {'trip_id', 'observation_id', 'arc_id', 'label', 'probability'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    df['trip_id'] = normalize_id(df['trip_id'])
    for c in ['observation_id', 'arc_id']:
        df[c] = pd.to_numeric(df[c], errors='raise').astype('int64')
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['probability'] = pd.to_numeric(df['probability'], errors='raise')
    if df['probability'].isna().any() or not df['probability'].between(0, 1).all():
        raise ValueError('Probabilities must be finite values in [0,1].')
    if df.duplicated(['trip_id', 'observation_id', 'arc_id']).any():
        raise ValueError('Duplicate candidate keys in validation predictions.')
    results = []
    per_trip = []
    for trip_id, trip in df.groupby('trip_id', sort=True):
        feats = load_features(trip_id)
        merged = trip.merge(feats, on=['observation_id', 'arc_id'], how='left', validate='one_to_one')
        if merged[['from_node_id', 'to_node_id']].isna().any().any():
            missing = int(merged[['from_node_id', 'to_node_id']].isna().any(axis=1).sum())
            raise ValueError(f'{trip_id}: {missing} candidates missing endpoint connectivity features')
        picks, disconnected = decode_trip(merged)
        for p in picks:
            results.append({'trip_id': trip_id, 'observation_id': int(p['observation_id']),
                            'arc_id': int(p['arc_id']), 'label': int(p['label']),
                            'probability': float(p['probability'])})
        per_trip.append({'trip_id': trip_id, 'observations': len(picks),
                         'disconnected_transitions': disconnected})
        print(f'{trip_id}: {len(picks)} observations, {disconnected} disconnected transitions')
    chosen = pd.DataFrame(results)
    chosen['hmm_selected'] = 1
    evaluated = df.merge(chosen[['trip_id','observation_id','arc_id','hmm_selected']],
                         on=['trip_id','observation_id','arc_id'], how='left', validate='one_to_one')
    evaluated['hmm_selected'] = evaluated['hmm_selected'].fillna(0).astype(int)
    # Baseline is the same top-1 selection policy, without sequential decoding.
    top_idx = df.groupby(['trip_id','observation_id'])['probability'].idxmax()
    top_keys = df.loc[top_idx, ['trip_id','observation_id','arc_id']].copy()
    top_keys['baseline_selected'] = 1
    evaluated = evaluated.merge(top_keys, on=['trip_id','observation_id','arc_id'], how='left', validate='one_to_one')
    evaluated['baseline_selected'] = evaluated['baseline_selected'].fillna(0).astype(int)
    before = binary_metrics(evaluated['label'], evaluated['baseline_selected'])
    after = binary_metrics(evaluated['label'], evaluated['hmm_selected'])
    baseline_selected = evaluated.loc[evaluated['baseline_selected'].eq(1)]
    hmm_selected = evaluated.loc[evaluated['hmm_selected'].eq(1)]
    report = {
        'warning': 'Validation uses route-derived pseudo-labels, not verified pointwise ground truth. This standalone decoder may differ from the student-test HMM script.',
        'validation_trips': int(df['trip_id'].nunique()),
        'validation_observations': int(chosen.shape[0]),
        'candidate_rows': int(len(df)),
        'baseline_top1_pseudo_label_agreement': float(baseline_selected['label'].mean()),
        'hmm_top1_pseudo_label_agreement': float(hmm_selected['label'].mean()),
        'baseline_one_selected_arc_per_observation': before,
        'hmm_one_selected_arc_per_observation': after,
        'hmm_disconnected_transitions': int(sum(x['disconnected_transitions'] for x in per_trip)),
    }
    (OUT/'validation_metrics.json').write_text(json.dumps(report, indent=2))
    chosen.to_csv(OUT/'validation_observation_matches.csv', index=False)
    pd.DataFrame(per_trip).to_csv(OUT/'validation_diagnostics.csv', index=False)
    print('\n=== BEFORE vs AFTER VITERBI (pseudo-labels) ===')
    print(f"Top-1 agreement: {report['baseline_top1_pseudo_label_agreement']:.2%} -> {report['hmm_top1_pseudo_label_agreement']:.2%}")
    for name in ['accuracy','precision','recall','f1']:
        print(f'{name}: {before[name]:.2%} -> {after[name]:.2%}')
    print(f"Disconnected transitions: {report['hmm_disconnected_transitions']}")
    print('Saved to:', OUT)


if __name__ == '__main__':
    main()
