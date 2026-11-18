"""Inference and server-side evaluation using the checked-in PathSnap HMM."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from PathSnap_Full_Route_Reconstruction import (
    load_network,
    make_geojson,
    reconstruct,
)
from PathSnap_Standalone_HMM_Evaluated import decode_trip, standalone_emissions
from backend.preprocessing import generate_hmm_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "artifacts" / "standalone_hmm_config.json"


class UnknownTripError(Exception):
    """Raised when a requested ID is not in the active partition."""


class PathSnapService:
    def __init__(self, root: Path = ROOT, config_path: Path = CONFIG_PATH, partition: str | None = None) -> None:
        self.root = root
        self.partition = partition or os.getenv("PATHSNAP_PARTITION") or (
            "evaluator_test" if (root / "evaluator_test").is_dir() else "student_test"
        )
        if self.partition not in {"student_test", "evaluator_test"}:
            raise ValueError("PATHSNAP_PARTITION must be student_test or evaluator_test")
        self.trip_root = root / "evaluator_test" if self.partition == "evaluator_test" else root / "Student folder" / "student_test"
        if not self.trip_root.is_dir():
            raise FileNotFoundError(f"Trip directory not found: {self.trip_root}")
        if not config_path.is_file():
            raise FileNotFoundError(f"HMM configuration not found: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        required_config = {
            "distance_sigma",
            "heading_sigma",
            "heading_weight",
            "max_hops",
            "disconnect_penalty",
            "hop_penalty",
            "reset_gap",
            "max_expansions",
        }
        missing_config = sorted(required_config - set(self.config))
        if missing_config:
            raise ValueError(f"HMM configuration is missing: {missing_config}")
        if self.config["distance_sigma"] <= 0 or self.config["heading_sigma"] <= 0:
            raise ValueError("HMM distance and heading sigmas must be positive.")
        if (
            self.config["heading_weight"] < 0
            or self.config["max_hops"] < 1
            or self.config["disconnect_penalty"] < 0
            or self.config["hop_penalty"] < 0
            or self.config["reset_gap"] < 1
            or self.config["max_expansions"] < 1
        ):
            raise ValueError("HMM decoding configuration contains invalid values.")

        # Discover paths only. Never inspect evaluator observations or labels at startup.
        self.trip_ids = {
            path.name for path in self.trip_root.iterdir()
            if path.is_dir() and not path.is_symlink() and re.fullmatch(r"[A-Za-z0-9_-]+", path.name)
            and all((path / f"{path.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))
        }

    def list_trips(self) -> list[dict[str, Any]]:
        return [
            {
                "trip_id": trip_id,
                "partition": self.partition,
            }
            for trip_id in sorted(self.trip_ids)
        ]

    def predict_trip(self, trip_id: str) -> dict[str, Any]:
        if trip_id not in self.trip_ids:
            raise UnknownTripError(f"Unknown {self.partition} trip: {trip_id}")

        feature_dir = self.root / "preprocessing_outputs" / self.partition
        trip_dir = self.trip_root / trip_id
        candidates_path = feature_dir / f"{trip_id}_candidates.parquet"
        observations_path = feature_dir / f"{trip_id}_observations.parquet"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = trip_dir / f"{trip_id}.route"
        for path in (nodes_path, arcs_path, trip_dir / f"{trip_id}.track"):
            if not path.is_file():
                raise FileNotFoundError(f"Required trip data not found: {path}")

        if self.partition == "student_test" and candidates_path.is_file() and observations_path.is_file():
            candidates = pd.read_parquet(candidates_path)
            observations = pd.read_parquet(observations_path)
        else:
            # Generate features in memory; no reference route or evaluator cache is used.
            observations, candidates = generate_hmm_features(trip_dir)
        observations = observations.sort_values("observation_id").reset_index(drop=True)
        required_features = {
            "observation_id",
            "arc_id",
            "from_node_id",
            "to_node_id",
            "distance_to_arc_m",
            "heading_difference_deg",
        }
        missing_features = sorted(required_features - set(candidates.columns))
        if missing_features:
            raise ValueError(
                f"Trip {trip_id} candidate features are missing: {missing_features}"
            )
        if candidates.empty:
            raise ValueError(f"Trip {trip_id} has no candidate roads.")
        if candidates.duplicated(["observation_id", "arc_id"]).any():
            raise ValueError(f"Trip {trip_id} contains duplicate candidate keys.")

        scored_candidates = candidates.copy()
        scored_candidates["probability"] = standalone_emissions(
            scored_candidates,
            sigma_distance=float(self.config["distance_sigma"]),
            sigma_heading=float(self.config["heading_sigma"]),
            heading_weight=float(self.config["heading_weight"]),
        )
        decoded, decode_stats = decode_trip(
            scored_candidates,
            max_hops=int(self.config["max_hops"]),
            disconnect_penalty=float(self.config["disconnect_penalty"]),
            hop_penalty=float(self.config["hop_penalty"]),
            reset_gap=int(self.config["reset_gap"]),
        )
        if decoded.empty:
            raise ValueError(f"No observations could be decoded for trip {trip_id}.")

        selected = decoded.merge(
            candidates,
            on=["observation_id", "arc_id"],
            how="left",
            validate="one_to_one",
        )
        if selected[["from_node_id", "to_node_id"]].isna().any().any():
            raise ValueError(f"Decoded road arcs could not be joined to candidates for {trip_id}.")

        coordinates, edges, graph = load_network(trip_dir, trip_id)
        route, unresolved_gaps, reconstruction_stats = reconstruct(
            selected,
            edges,
            graph,
            int(self.config["max_expansions"]),
        )
        route_geojson = make_geojson(route, coordinates, edges)
        # Show actual nearby candidate roads without exposing the reference route.
        candidate_arc_ids = sorted(set(candidates["arc_id"].astype(int)) & set(edges))
        roads_geojson = make_geojson(
            [(arc_id, False) for arc_id in candidate_arc_ids], coordinates, edges
        )

        # Reference data is read only after inference and route reconstruction.
        route_membership_rate = None
        if route_path.is_file():
            route_ids = pd.to_numeric(
                pd.read_csv(route_path, sep=r"\s+", header=None).iloc[:, 0],
                errors="coerce",
            ).dropna().astype("int64").unique()
            route_membership_rate = float(selected["arc_id"].astype(int).isin(set(route_ids.tolist())).mean())

        gps_track = [
            {
                "observation_id": int(row.observation_id),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }
            for row in observations.itertuples(index=False)
        ]
        predictions = [
            {
                "observation_id": int(row.observation_id),
                "arc_id": int(row.arc_id),
                "emission_probability": float(row.probability),
                "transition_hops": int(row.transition_hops),
                "disconnected_transition": bool(row.disconnected_transition),
            }
            for row in decoded.itertuples(index=False)
        ]
        transition_count = max(0, len(route) - 1)

        return {
            "trip_id": trip_id,
            "partition": self.partition,
            "metrics": {
                "total_observations": len(observations),
                "observations_with_candidates": int(
                    candidates["observation_id"].nunique()
                ),
                "decoded_observations": int(decode_stats["observations"]),
                "candidate_coverage": (
                    candidates["observation_id"].nunique() / len(observations)
                    if len(observations)
                    else 0.0
                ),
                "prediction_coverage": (
                    decode_stats["observations"] / len(observations)
                    if len(observations)
                    else 0.0
                ),
                "route_membership_proxy_rate": route_membership_rate,
                "route_membership_proxy_note": (
                    "Fraction of predicted arcs occurring anywhere in the reference "
                    "route; not observation-level accuracy."
                ),
                "candidate_decoder_disconnected_transitions": int(
                    decode_stats["disconnected_transitions"]
                ),
                "full_network_bridged_transitions": int(
                    reconstruction_stats["bridged_transitions"]
                ),
                "inserted_intermediate_arcs": int(
                    reconstruction_stats["inserted_intermediate_arcs"]
                ),
                "unresolved_route_gaps": len(unresolved_gaps),
                "connected_transition_rate": (
                    1 - len(unresolved_gaps) / transition_count
                    if transition_count
                    else 1.0
                ),
            },
            "gps_track": gps_track,
            "road_network_geojson": roads_geojson,
            "predictions": predictions,
            "predicted_route_geojson": route_geojson,
            "unresolved_gaps": unresolved_gaps,
        }
