"""Inference and server-side evaluation using the checked-in PathSnap HMM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from PathSnap_Full_Route_Reconstruction import (
    load_network,
    make_geojson,
    reconstruct,
)
from PathSnap_Standalone_HMM_Evaluated import decode_trip, standalone_emissions


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "artifacts" / "standalone_hmm_config.json"


class UnknownTripError(Exception):
    """Raised when a requested ID is not in the reserved student-test set."""


class PathSnapService:
    def __init__(self, root: Path = ROOT, config_path: Path = CONFIG_PATH) -> None:
        self.root = root
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

        manifest_path = root / "preprocessing_outputs" / "processing_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Preprocessing manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path, dtype={"trip_id": str})
        self.trips = manifest.loc[
            (manifest["partition"] == "student_test")
            & (manifest["status"] == "ok"),
            ["trip_id", "valid_observations", "candidate_rows"],
        ].fillna(0)
        self.trip_ids = set(self.trips["trip_id"].astype(str))

    def list_trips(self) -> list[dict[str, Any]]:
        return [
            {
                "trip_id": str(row.trip_id),
                "observations": int(row.valid_observations),
                "candidate_rows": int(row.candidate_rows),
            }
            for row in self.trips.itertuples(index=False)
        ]

    def predict_trip(self, trip_id: str) -> dict[str, Any]:
        if trip_id not in self.trip_ids:
            raise UnknownTripError(f"Unknown student-test trip: {trip_id}")

        feature_dir = self.root / "preprocessing_outputs" / "student_test"
        trip_dir = self.root / "Student folder" / "student_test" / trip_id
        candidates_path = feature_dir / f"{trip_id}_candidates.parquet"
        observations_path = feature_dir / f"{trip_id}_observations.parquet"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = trip_dir / f"{trip_id}.route"
        for path in (candidates_path, observations_path, nodes_path, arcs_path, route_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required trip data not found: {path}")

        candidates = pd.read_parquet(candidates_path)
        observations = pd.read_parquet(observations_path).sort_values(
            "observation_id"
        ).reset_index(drop=True)
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

        # Reference data is read only after inference and route reconstruction.
        route_ids = pd.to_numeric(
            pd.read_csv(route_path, sep=r"\s+", header=None).iloc[:, 0],
            errors="coerce",
        ).dropna().astype("int64").unique()
        route_set = set(route_ids.tolist())
        route_membership = selected["arc_id"].astype(int).isin(route_set)

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
                "route_membership_proxy_rate": float(route_membership.mean()),
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
            "predictions": predictions,
            "predicted_route_geojson": route_geojson,
            "unresolved_gaps": unresolved_gaps,
        }
