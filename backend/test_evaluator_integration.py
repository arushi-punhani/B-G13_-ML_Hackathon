"""Integration tests using student-test data copied into temporary fixtures only."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.preprocessing import generate_hmm_features
from backend.service import PathSnapService, UnknownTripError

ROOT = Path(__file__).resolve().parents[1]
TID = '00000052'


class EvaluatorIntegrationTests(unittest.TestCase):
    def test_on_demand_features_match_existing_preprocessing(self):
        source = ROOT / 'Student folder' / 'student_test' / TID
        obs, candidates = generate_hmm_features(source)
        expected = pd.read_parquet(ROOT / 'preprocessing_outputs' / 'student_test' / f'{TID}_candidates.parquet')
        expected_obs = pd.read_parquet(ROOT / 'preprocessing_outputs' / 'student_test' / f'{TID}_observations.parquet')
        self.assertEqual(obs.observation_id.tolist(), expected_obs.observation_id.tolist())
        for column in ['observation_id','arc_id','from_node_id','to_node_id','candidate_rank']:
            np.testing.assert_array_equal(candidates[column], expected[column])
        for column in ['distance_to_arc_m','heading_difference_deg']:
            np.testing.assert_allclose(candidates[column], expected[column], equal_nan=True)

    def test_evaluator_discovery_and_inference_without_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / 'evaluator_test' / TID
            dest.mkdir(parents=True)
            source = ROOT / 'Student folder' / 'student_test' / TID
            # Deliberately omit .route: evaluator inference must work without labels.
            for ext in ('track', 'nodes', 'arcs'):
                shutil.copyfile(source / f'{TID}.{ext}', dest / f'{TID}.{ext}')
            service = PathSnapService(root=root, partition='evaluator_test')
            self.assertEqual(service.list_trips(), [{'trip_id': TID, 'partition': 'evaluator_test'}])
            result = service.predict_trip(TID)
            self.assertEqual(result['partition'], 'evaluator_test')
            self.assertEqual(result['metrics']['total_observations'], 262)
            self.assertIsNone(result['metrics']['route_membership_proxy_rate'])
            self.assertTrue(result['predicted_route_geojson']['features'])
            self.assertTrue(result['road_network_geojson']['features'])
            self.assertNotIn('reference_route', result)
            with self.assertRaises(UnknownTripError):
                service.predict_trip('../00000052')

    def test_student_partition_remains_available(self):
        service = PathSnapService(root=ROOT, partition='student_test')
        self.assertEqual(len(service.list_trips()), 10)
        self.assertTrue(all(t['partition'] == 'student_test' for t in service.list_trips()))


if __name__ == '__main__':
    unittest.main()
