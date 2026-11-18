"""Generate the HMM's candidate features using the existing EDA pipeline settings.

Only track, nodes and arcs are inputs. Reference routes are never opened here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from shapely import STRtree


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+", header=None)
    if table.shape[1] != len(names):
        raise ValueError(f"{path}: expected {len(names)} columns, got {table.shape[1]}")
    table.columns = names
    return table.apply(pd.to_numeric, errors="coerce")


def _valid_coords(frame: pd.DataFrame) -> pd.Series:
    return (np.isfinite(frame.longitude) & np.isfinite(frame.latitude)
            & frame.longitude.between(-180, 180) & frame.latitude.between(-90, 90))


def generate_hmm_features(folder: Path, radius_m: float = 100.0, max_candidates: int = 8):
    """Return observations and candidate frames; preserves raw 0-based IDs."""
    tid = folder.name
    track = _read(folder / f"{tid}.track", ["longitude", "latitude", "timestamp"])
    nodes = _read(folder / f"{tid}.nodes", ["longitude", "latitude"])
    arcs = _read(folder / f"{tid}.arcs", ["from_node_id", "to_node_id"])
    track.insert(0, "observation_id", np.arange(len(track)))
    gps = track.loc[_valid_coords(track) & np.isfinite(track.timestamp)].copy().reset_index(drop=True)
    if gps.empty:
        raise ValueError(f"Trip {tid} has no valid GPS observations")
    valid_nodes = _valid_coords(nodes).to_numpy()
    raw = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    valid = np.isfinite(raw).all(axis=1) & (raw == np.floor(raw)).all(axis=1)
    valid &= ((raw >= 0) & (raw < len(nodes))).all(axis=1)
    positions = np.flatnonzero(valid)
    refs = raw[positions].astype(np.int64)
    valid[positions] &= valid_nodes[refs[:, 0]] & valid_nodes[refs[:, 1]]
    arcs = arcs.loc[valid].copy()
    arcs.insert(0, "arc_id", arcs.index.to_numpy())
    arcs[["from_node_id", "to_node_id"]] = arcs[["from_node_id", "to_node_id"]].astype("int64")
    if arcs.empty:
        raise ValueError(f"Trip {tid} has no valid road arcs")
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={float(gps.latitude.median())} +lon_0={float(gps.longitude.median())} +datum=WGS84 +units=m")
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gx, gy = map(np.asarray, transform.transform(gps.longitude.to_numpy(), gps.latitude.to_numpy()))
    if not (np.isfinite(gx).all() and np.isfinite(gy).all()):
        raise ValueError(f"Trip {tid} GPS projection failed")
    adjacent = gps.observation_id.diff().eq(1).to_numpy()
    dt = gps.timestamp.diff().to_numpy()
    dx, dy = np.r_[np.nan, np.diff(gx)], np.r_[np.nan, np.diff(gy)]
    movement = adjacent & (dt > 0) & np.isfinite(dt) & (np.hypot(dx, dy) > 0)
    heading = np.where(movement, (np.degrees(np.arctan2(dx, dy)) + 360) % 360, np.nan)
    lon, lat = nodes.longitude.to_numpy(), nodes.latitude.to_numpy()
    u, v = arcs.from_node_id.to_numpy(), arcs.to_node_id.to_numpy()
    ux, uy = transform.transform(lon[u], lat[u])
    vx, vy = transform.transform(lon[v], lat[v])
    ux, uy, vx, vy = map(np.asarray, (ux, uy, vx, vy))
    finite = np.isfinite(ux) & np.isfinite(uy) & np.isfinite(vx) & np.isfinite(vy)
    arcs = arcs.loc[finite].reset_index(drop=True)
    ux, uy, vx, vy = ux[finite], uy[finite], vx[finite], vy[finite]
    if arcs.empty:
        raise ValueError(f"Trip {tid} has no projectable road arcs")
    lines = shapely.linestrings(np.stack([np.column_stack([ux, uy]), np.column_stack([vx, vy])], axis=1))
    points = shapely.points(gx, gy)
    pi, ai = STRtree(lines).query(points, predicate="dwithin", distance=radius_m)
    distance = shapely.distance(points[pi], lines[ai])
    order = np.lexsort((arcs.arc_id.to_numpy()[ai], distance, pi))
    pi, ai, distance = pi[order], ai[order], distance[order]
    ranks = pd.Series(pi).groupby(pi, sort=False).cumcount().to_numpy() + 1
    keep = ranks <= max_candidates
    pi, ai, distance, ranks = pi[keep], ai[keep], distance[keep], ranks[keep]
    edge_dx, edge_dy = vx[ai] - ux[ai], vy[ai] - uy[ai]
    length = np.hypot(edge_dx, edge_dy)
    bearing = np.where(length > 0, (np.degrees(np.arctan2(edge_dx, edge_dy)) + 360) % 360, np.nan)
    angle = np.abs((heading[pi] - bearing + 180) % 360 - 180)
    candidates = pd.DataFrame({"observation_id": gps.observation_id.to_numpy()[pi],
        "arc_id": arcs.arc_id.to_numpy()[ai], "from_node_id": arcs.from_node_id.to_numpy()[ai],
        "to_node_id": arcs.to_node_id.to_numpy()[ai], "candidate_rank": ranks,
        "distance_to_arc_m": distance, "heading_difference_deg": angle})
    return gps, candidates
