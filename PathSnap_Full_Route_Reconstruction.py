#!/usr/bin/env python3
"""Reconstruct a directed road route from HMM observation-to-arc matches.

Run from the PathSnap project root:
    python PathSnap_Full_Route_Reconstruction.py --model mlp
    python PathSnap_Full_Route_Reconstruction.py --model standalone --partition train

Uses the full per-trip .arcs and .nodes road network, NOT candidate-only arcs.
Generates per-trip route_arc_ids.csv, route.geojson, gps_vs_route.html
(if folium is installed), diagnostics.csv and a short report-ready summary.
"""
import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def haversine(lon1, lat1, lon2, lat2):
    r = 6371008.8
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = a2 - a1, math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(dlon/2)**2
    return r * 2 * math.asin(min(1, math.sqrt(max(0, a))))


def load_network(folder, tid):
    nodes = pd.read_csv(folder / f'{tid}.nodes', sep=r'\s+', header=None, names=['lon','lat'])
    arcs = pd.read_csv(folder / f'{tid}.arcs', sep=r'\s+', header=None, names=['u','v'])
    coords = nodes[['lon','lat']].to_numpy(dtype=float)
    uv = arcs[['u','v']].to_numpy(dtype=float)
    valid = np.isfinite(uv).all(axis=1) & np.all(uv == np.floor(uv), axis=1)
    valid &= (uv >= 0).all(axis=1) & (uv < len(coords)).all(axis=1)
    ids = np.flatnonzero(valid)
    if len(ids):
        a = uv[ids].astype(np.int64)
        valid[ids] &= np.isfinite(coords[a]).all(axis=(1,2))
    graph = defaultdict(list)
    edges = {}
    for arc_id in np.flatnonzero(valid):
        u, v = map(int, uv[arc_id])
        length = haversine(*coords[u], *coords[v])
        edges[int(arc_id)] = (u, v, length)
        graph[u].append((v, length, int(arc_id)))
    return coords, edges, graph


def shortest_path(graph, start, goal, max_expansions=200000):
    """Dijkstra over directed full network. Returns (arc IDs, length), or None."""
    if start == goal:
        return [], 0.0
    heap = [(0.0, start)]
    dist = {start: 0.0}
    prev = {}
    expanded = 0
    while heap and expanded < max_expansions:
        cost, u = heapq.heappop(heap)
        if cost > dist[u] + 1e-9:
            continue
        if u == goal:
            path = []
            while u != start:
                parent, arc_id = prev[u]
                path.append(arc_id)
                u = parent
            path.reverse()
            return path, cost
        expanded += 1
        for v, length, arc_id in graph.get(u, []):
            alt = cost + length
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                prev[v] = (u, arc_id)
                heapq.heappush(heap, (alt, v))
    return None


def reconstruct(matches, edges, graph, max_expansions):
    """Bridge each change of selected arc; preserve gaps rather than invent roads."""
    matched = matches.sort_values('observation_id').drop_duplicates('observation_id')
    route = []
    gaps = []
    bridged = 0
    inserted = 0
    missing = 0
    previous_arc = None
    for row in matched.itertuples(index=False):
        arc_id = int(row.arc_id)
        if arc_id not in edges:
            missing += 1
            gaps.append({'observation_id':int(row.observation_id), 'reason':'selected_arc_not_in_valid_network', 'previous_arc_id':previous_arc, 'next_arc_id':arc_id})
            previous_arc = None
            continue
        if previous_arc is None:
            route.append((arc_id, False))
        elif arc_id == previous_arc:
            continue
        else:
            end_prev = edges[previous_arc][1]
            start_next = edges[arc_id][0]
            if end_prev != start_next:
                bridge = shortest_path(graph, end_prev, start_next, max_expansions)
                if bridge is None:
                    gaps.append({'observation_id':int(row.observation_id), 'reason':'no_directed_path', 'previous_arc_id':previous_arc, 'next_arc_id':arc_id})
                else:
                    bridge_ids, _ = bridge
                    route.extend((a, True) for a in bridge_ids)
                    inserted += len(bridge_ids)
                    bridged += 1
            route.append((arc_id, False))
        previous_arc = arc_id
    return route, gaps, {'selected_observations':len(matched), 'missing_arc_observations':missing,
                         'bridged_transitions':bridged, 'inserted_intermediate_arcs':inserted,
                         'unresolved_gaps':len(gaps), 'reconstructed_arc_entries':len(route)}


def make_geojson(route, coords, edges):
    features = []
    # Individual arc features preserve gaps; no fabricated straight connections.
    for index, (arc_id, is_bridge) in enumerate(route):
        u, v, length = edges[arc_id]
        features.append({'type':'Feature',
                         'properties':{'sequence':index,'arc_id':arc_id,'intermediate_bridge':is_bridge,'length_m':round(length,2)},
                         'geometry':{'type':'LineString','coordinates':[coords[u].tolist(),coords[v].tolist()]}})
    return {'type':'FeatureCollection','features':features}


def make_map(folder, tid, geojson, destination):
    try:
        import folium
    except ImportError:
        return False
    track_path = folder / f'{tid}.track'
    gps = pd.read_csv(track_path, sep=r'\s+', header=None, usecols=[0,1], names=['lon','lat'])
    gps = gps.replace([np.inf,-np.inf],np.nan).dropna()
    gps = gps[gps.lon.between(-180,180) & gps.lat.between(-90,90)]
    features = geojson['features']
    if gps.empty and not features:
        return False
    if not gps.empty:
        center = [float(gps.lat.median()),float(gps.lon.median())]
    else:
        first = features[0]['geometry']['coordinates'][0]
        center = [first[1],first[0]]
    m = folium.Map(location=center, zoom_start=14, tiles='OpenStreetMap')
    if not gps.empty:
        points = gps.iloc[::max(1,len(gps)//1200)]
        folium.PolyLine(points[['lat','lon']].values.tolist(),color='#e11d48',weight=2,opacity=0.65,
                        tooltip='Raw GPS track (sampled)').add_to(m)
    main = [f for f in features if not f['properties']['intermediate_bridge']]
    bridge = [f for f in features if f['properties']['intermediate_bridge']]
    for group,color,label in [(main,'#2563eb','HMM selected arcs'),(bridge,'#16a34a','Shortest-path intermediate arcs')]:
        if group:
            folium.GeoJson({'type':'FeatureCollection','features':group},name=label,
                           style_function=lambda feature, c=color:{'color':c,'weight':4,'opacity':0.85}).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(destination))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',choices=['mlp','standalone'],default='mlp')
    ap.add_argument('--partition',choices=['train','student_test'],default='student_test')
    ap.add_argument('--matches',type=Path,help='Optional path to observation_matches.csv')
    ap.add_argument('--trip',help='Optional single 8-digit trip ID')
    ap.add_argument('--max-expansions',type=int,default=200000)
    ap.add_argument('--no-map',action='store_true')
    args = ap.parse_args()
    if args.max_expansions < 1:
        ap.error('--max-expansions must be positive')
    if args.matches:
        source = args.matches
    elif args.model == 'mlp':
        source = ROOT / 'hmm_outputs' / 'mlp' / 'observation_matches.csv'
    else:
        sub = 'standalone_train_evaluated' if args.partition == 'train' else 'standalone_student_test'
        source = ROOT / 'hmm_outputs' / sub / 'observation_matches.csv'
    if not source.exists():
        ap.error(f'Predictions missing: {source}. Run the HMM decoder first.')
    matches = pd.read_csv(source,dtype={'trip_id':str})
    if not {'trip_id','observation_id','arc_id'}.issubset(matches.columns):
        ap.error('Matches must contain trip_id, observation_id, arc_id')
    matches['trip_id'] = matches.trip_id.str.zfill(8)
    if args.trip:
        matches = matches.loc[matches.trip_id.eq(args.trip.zfill(8))]
    if matches.empty:
        ap.error('No matching observations found')
    output = ROOT / 'hmm_outputs' / ('full_route_' + args.model + '_' + args.partition)
    output.mkdir(parents=True,exist_ok=True)
    diagnostics = []
    for tid,trip_matches in matches.groupby('trip_id',sort=True):
        folder = ROOT / 'Student folder' / args.partition / tid
        if not folder.is_dir():
            print(f'{tid}: raw data folder missing: {folder}; skipped')
            continue
        coords,edges,graph = load_network(folder,tid)
        route,gaps,stats = reconstruct(trip_matches,edges,graph,args.max_expansions)
        trip_out = output / tid
        trip_out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame([{'sequence':i,'arc_id':arc,'is_intermediate_bridge':bridge,
                       'from_node_id':edges[arc][0],'to_node_id':edges[arc][1],
                       'arc_length_m':edges[arc][2]} for i,(arc,bridge) in enumerate(route)],
                     columns=['sequence','arc_id','is_intermediate_bridge','from_node_id','to_node_id','arc_length_m']).to_csv(trip_out/'route_arc_ids.csv',index=False)
        pd.DataFrame(gaps,columns=['observation_id','reason','previous_arc_id','next_arc_id']).to_csv(trip_out/'unresolved_gaps.csv',index=False)
        geojson = make_geojson(route,coords,edges)
        (trip_out/'route.geojson').write_text(json.dumps(geojson),encoding='utf-8')
        mapped = False if args.no_map else make_map(folder,tid,geojson,trip_out/'gps_vs_route.html')
        stats.update({'trip_id':tid,'road_network_arcs':len(edges),'map_created':mapped,
                      'route_length_m':round(sum(edges[a][2] for a,_ in route),1)})
        diagnostics.append(stats)
        print(f'{tid}: {stats["reconstructed_arc_entries"]} route arcs, '
              f'{stats["inserted_intermediate_arcs"]} inserted, '
              f'{stats["unresolved_gaps"]} unresolved gaps, map={mapped}',flush=True)
    if not diagnostics:
        ap.error('No trips processed. Check Student folder/<partition>/<trip_id> directory structure.')
    df = pd.DataFrame(diagnostics)
    df.to_csv(output/'route_reconstruction_diagnostics.csv',index=False)
    summary = f'''# PathSnap full-route reconstruction summary

Input: {source}
Partition: {args.partition}
Trips processed: {len(df)}
GPS observations decoded: {int(df.selected_observations.sum()):,}
Reconstructed arc entries: {int(df.reconstructed_arc_entries.sum()):,}
Intermediate road arcs inserted: {int(df.inserted_intermediate_arcs.sum()):,}
Unresolved directed-network gaps: {int(df.unresolved_gaps.sum()):,}
Total reconstructed arc length: {df.route_length_m.sum()/1000:.2f} km (may include repeated traversals)

## Method
Each Viterbi-selected road arc is preserved in observation order. For each change
of selected arc, the decoder finds a minimum-length directed path through the full
trip-specific road network from the previous arc's endpoint to the next arc's
start node using Dijkstra's algorithm and haversine edge weights. Intermediate
arcs are inserted in order. If no directed path exists, the gap is reported
instead of drawing a false connection. GeoJSON retains each arc individually.

## Limitations
This is a postprocessing reconstruction, not a joint full-network HMM decoder.
Shortest path is a distance-based heuristic; it may not be the true driven route.
A gap-free output is not proof of ground-truth correctness. GPS track and
reconstructed route visualizations are qualitative, not accuracy measurements.
Existing classifier/HMM scores are against route-derived pseudo-labels.
'''
    (output/'hmm_report_summary.md').write_text(summary,encoding='utf-8')
    print(f'\nSaved full-route reconstruction to: {output}')
    print('See per-trip gps_vs_route.html (install folium if absent), route.geojson, and diagnostics.csv.')

if __name__ == '__main__':
    main()
