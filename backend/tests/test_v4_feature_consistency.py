import csv
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from evaluation.build_v4_rich_features import extract_rich_features
from services.fusion_service import AdaptiveScoreFusion
from services.retrieval_feature_service import (
    RetrievalFeatureService,
    V4_RICH_FEATURES,
    V4_RICH_FEATURE_TOP_K,
    retrieval_feature_service,
)


class StubModel:
    def __init__(self, prediction=0.5):
        self.prediction = prediction
        self.last_frame = None

    def predict(self, frame):
        self.last_frame = frame
        return np.array([self.prediction])


stub_model = StubModel()
with patch.object(joblib, "load", return_value=stub_model):
    with patch("os.path.exists", return_value=True):
        predictor_module = importlib.import_module(
            "services.adaptive_predictor_v4"
        )

AdaptivePredictorV4 = predictor_module.AdaptivePredictorV4
adaptive_predictor_v4 = predictor_module.adaptive_predictor_v4


class V4FeatureConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.query = "How do neural retrieval models rank documents?"
        self.bm25_docs = [
            "neural retrieval ranks documents",
            "document ranking methods",
            "retrieval models compare passages",
        ]
        self.dense_docs = [
            "neural retrieval ranks documents",
            "semantic search over text",
            "document ranking methods",
        ]
        self.bm25_scores = [8.0, 5.0, 2.0]
        self.dense_distances = [0.1, 0.4, 0.7]
        self.bm25_ids = ["doc-a", "doc-b", "doc-c"]
        self.dense_ids = ["doc-a", "doc-d", "doc-b"]

    def extract_features(self, top_k=None):
        service = (
            retrieval_feature_service
            if top_k is None
            else RetrievalFeatureService(top_k=top_k)
        )
        return service.extract(
            query=self.query,
            bm25_scores=self.bm25_scores,
            dense_distances=self.dense_distances,
            bm25_docs=self.bm25_docs,
            dense_docs=self.dense_docs,
            bm25_ids=self.bm25_ids,
            dense_ids=self.dense_ids,
        )

    def test_feature_schema_order_and_training_dimension_match(self):
        features = self.extract_features()

        trainer_path = (
            Path(__file__).resolve().parents[1]
            / "training"
            / "train_v4_rich_model.py"
        )
        self.assertIn(
            "FEATURES = list(V4_RICH_FEATURES)",
            trainer_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(list(features)[:17], list(V4_RICH_FEATURES))
        self.assertEqual(len(features), 17 + 8)
        self.assertEqual(len([features[name] for name in V4_RICH_FEATURES]), 17)
        self.assertEqual(
            tuple(predictor_module.adaptive_predictor_v4.features),
            V4_RICH_FEATURES,
        )
        self.assertTrue(
            all(
                isinstance(features[name], (int, float, np.number))
                for name in V4_RICH_FEATURES
            )
        )
        training_dir = Path(__file__).resolve().parents[1]
        with (training_dir / "evaluation" / "v4_analysis" / "rich_features.csv").open(
            encoding="utf-8",
            newline="",
        ) as file:
            training_header = next(csv.reader(file))
        training_metadata = json.loads(
            (
                training_dir
                / "evaluation"
                / "v4_analysis"
                / "v4_rich_model_metadata.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(training_header[1:18], list(V4_RICH_FEATURES))
        self.assertEqual(training_metadata["features"], list(V4_RICH_FEATURES))

    def test_builder_and_inference_use_identical_feature_values(self):
        inference = self.extract_features()
        training = extract_rich_features(
            query=self.query,
            bm25_scores=self.bm25_scores,
            dense_distances=self.dense_distances,
            bm25_docs=self.bm25_docs,
            dense_docs=self.dense_docs,
            bm25_ids=self.bm25_ids,
            dense_ids=self.dense_ids,
        )

        self.assertEqual(tuple(training), V4_RICH_FEATURES)
        for name in V4_RICH_FEATURES:
            self.assertAlmostEqual(training[name], inference[name])
        self.assertAlmostEqual(inference["overlap_ratio"], 0.5)
        self.assertAlmostEqual(inference["rank_agreement"], 1.0 - 0.5 / 9.0)

    def test_query_dependent_features_use_the_supplied_query(self):
        features = self.extract_features()
        self.assertGreater(features["query_token_count"], 0)
        self.assertGreater(features["query_unique_token_count"], 0)
        self.assertGreater(features["query_avg_token_length"], 0)
        self.assertGreater(features["bm25_top_lexical_coverage"], 0)
        self.assertGreater(features["dense_top_lexical_coverage"], 0)

        other = retrieval_feature_service.extract(
            query="unrelated vocabulary",
            bm25_scores=self.bm25_scores,
            dense_distances=self.dense_distances,
            bm25_docs=self.bm25_docs,
            dense_docs=self.dense_docs,
            bm25_ids=self.bm25_ids,
            dense_ids=self.dense_ids,
        )
        self.assertNotEqual(
            features["query_token_count"],
            other["query_token_count"],
        )
        self.assertNotEqual(
            features["bm25_top_lexical_coverage"],
            other["bm25_top_lexical_coverage"],
        )

    def assert_builder_runtime_parity(
        self,
        query,
        bm25_scores,
        dense_distances,
        bm25_docs,
        dense_docs,
        bm25_ids,
        dense_ids,
    ):
        built = extract_rich_features(
            query,
            bm25_scores,
            dense_distances,
            bm25_docs,
            dense_docs,
            bm25_ids,
            dense_ids,
        )
        runtime = retrieval_feature_service.extract(
            query=query,
            bm25_scores=bm25_scores,
            dense_distances=dense_distances,
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            bm25_ids=bm25_ids,
            dense_ids=dense_ids,
        )
        self.assertEqual(tuple(built), V4_RICH_FEATURES)
        for feature in V4_RICH_FEATURES:
            self.assertAlmostEqual(
                built[feature],
                runtime[feature],
                places=12,
                msg=feature,
            )
        return runtime

    def test_parity_across_query_and_candidate_scenarios(self):
        cases = (
            {
                "name": "technical lexical",
                "query": "neural retrieval rank document",
                "bm25_scores": [10.0, 6.0],
                "dense_distances": [0.9, 1.1],
                "bm25_docs": [
                    "neural retrieval rank document",
                    "another technical result",
                ],
                "dense_docs": ["semantic passage", "another passage"],
                "bm25_ids": ["a", "a2"],
                "dense_ids": ["b", "b2"],
            },
            {
                "name": "conceptual dense",
                "query": "How can meaning be inferred across different wording?",
                "bm25_scores": [2.0, 1.0],
                "dense_distances": [0.1, 0.2],
                "bm25_docs": ["lexical unrelated", "finance literal phrase"],
                "dense_docs": [
                    "meaning inferred from alternative wording",
                    "a second semantic match",
                ],
                "bm25_ids": ["c", "c2"],
                "dense_ids": ["d", "d2"],
            },
            {
                "name": "overlapping identifiers",
                "query": "shared retrieval results",
                "bm25_scores": [5.0, 3.0, 1.0],
                "dense_distances": [0.1, 0.5, 0.9],
                "bm25_docs": ["one", "two", "three"],
                "dense_docs": ["three", "one", "four"],
                "bm25_ids": ["id-1", "id-2", "id-3"],
                "dense_ids": ["id-3", "id-1", "id-4"],
            },
            {
                "name": "no candidate overlap",
                "query": "disjoint candidate identifiers",
                "bm25_scores": [2.0, 1.0],
                "dense_distances": [0.2, 0.6],
                "bm25_docs": ["same text", "bm25 second"],
                "dense_docs": ["same text", "dense second"],
                "bm25_ids": ["bm-a", "bm-b"],
                "dense_ids": ["de-a", "de-b"],
            },
            {
                "name": "fewer than ten",
                "query": "small candidate pool",
                "bm25_scores": [4.0, 4.0, 1.0],
                "dense_distances": [0.25, 0.25, 0.75],
                "bm25_docs": ["pool first", "pool second", "pool third"],
                "dense_docs": ["pool second", "pool first", "pool third"],
                "bm25_ids": ["p1", "p2", "p3"],
                "dense_ids": ["p2", "p1", "p3"],
            },
            {
                "name": "equal scores and tied rank order",
                "query": "equal score tied ordering",
                "bm25_scores": [1.0, 1.0, 1.0],
                "dense_distances": [0.5, 0.5, 0.5],
                "bm25_docs": ["a", "b", "c"],
                "dense_docs": ["a", "b", "c"],
                "bm25_ids": ["t1", "t2", "t3"],
                "dense_ids": ["t1", "t2", "t3"],
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                features = self.assert_builder_runtime_parity(
                    **{
                        key: value
                        for key, value in case.items()
                        if key != "name"
                    }
                )
                if case["name"] == "no candidate overlap":
                    self.assertEqual(features["overlap_ratio"], 0.0)
                    self.assertEqual(features["rank_agreement"], 0.0)
                if case["name"] == "equal scores and tied rank order":
                    self.assertEqual(features["bm25_score_gap"], 0.0)
                    self.assertEqual(features["dense_similarity_gap"], 0.0)
                    self.assertEqual(features["rank_agreement"], 1.0)
                if case["name"] == "fewer than ten":
                    self.assertEqual(features["bm25_score_gap"], 0.0)

    def test_training_dense_similarity_transform_is_consistent(self):
        features = self.extract_features()
        expected = 1.0 - np.asarray(self.dense_distances)
        self.assertAlmostEqual(
            features["dense_top_similarity"],
            float(expected[0]),
        )
        self.assertAlmostEqual(
            features["dense_mean_similarity"],
            float(np.mean(expected)),
        )
        self.assertAlmostEqual(
            features["dense_similarity_std"],
            float(np.std(expected)),
        )
        self.assertAlmostEqual(
            features["dense_similarity_gap"],
            float(expected[0] - expected[1]),
        )
        self.assertAlmostEqual(
            features["dense_top_mean_ratio"],
            float(expected[0] / np.mean(expected)),
        )
        self.assertEqual(V4_RICH_FEATURE_TOP_K, 10)

    def test_benchmark_candidate_pool_and_feature_window_are_explicit(self):
        evaluator_path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "benchmark_v4_rich_actual.py"
        )
        evaluator_source = evaluator_path.read_text(encoding="utf-8")
        self.assertIn("CANDIDATE_K = 50", evaluator_source)
        self.assertIn('"feature_top_k": V4_RICH_FEATURE_TOP_K', evaluator_source)
        self.assertIn("top_k=CANDIDATE_K", evaluator_source)

    def test_feature_calculations_use_only_training_top_k_window(self):
        top_k = V4_RICH_FEATURE_TOP_K
        scores = list(range(100, 100 - top_k, -1))
        distances = [value / 100 for value in range(top_k)]
        documents = [f"query token document {index}" for index in range(top_k)]
        base = retrieval_feature_service.extract(
            query="query token",
            bm25_scores=scores,
            dense_distances=distances,
            bm25_docs=documents,
            dense_docs=documents,
            bm25_ids=[f"doc-{i}" for i in range(top_k)],
            dense_ids=[f"doc-{i}" for i in range(top_k)],
        )

        extended = retrieval_feature_service.extract(
            query="query token",
            bm25_scores=scores + [-10000],
            dense_distances=distances + [10000],
            bm25_docs=documents + ["extra token document"],
            dense_docs=documents + ["extra token document"],
            bm25_ids=[f"doc-{i}" for i in range(top_k + 1)],
            dense_ids=[f"doc-{i}" for i in range(top_k + 1)],
        )

        for name in V4_RICH_FEATURES:
            self.assertAlmostEqual(base[name], extended[name])

    def test_empty_query_fails_and_empty_retrieval_has_zero_semantics(self):
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            retrieval_feature_service.extract(
                query=" ",
                bm25_scores=[],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
                bm25_ids=[],
                dense_ids=[],
            )

        features = retrieval_feature_service.extract(
            query="query",
            bm25_scores=[],
            dense_distances=[],
            bm25_docs=[],
            dense_docs=[],
            bm25_ids=[],
            dense_ids=[],
        )
        for name in V4_RICH_FEATURES[:12] + V4_RICH_FEATURES[15:]:
            self.assertEqual(features[name], 0)
        self.assertEqual(features["query_token_count"], 1)
        self.assertEqual(features["query_unique_token_count"], 1)
        self.assertEqual(features["query_avg_token_length"], 5.0)

        punctuation_query = retrieval_feature_service.extract(
            query="?!",
            bm25_scores=[],
            dense_distances=[],
            bm25_docs=[],
            dense_docs=[],
            bm25_ids=[],
            dense_ids=[],
        )
        self.assertEqual(punctuation_query["query_token_count"], 0)
        self.assertEqual(punctuation_query["query_avg_token_length"], 0.0)

    def test_legacy_call_without_query_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Invalid positional arguments"):
            retrieval_feature_service.extract(
                self.bm25_scores,
                self.dense_distances,
                self.bm25_docs,
                self.dense_docs,
            )

    def test_missing_inputs_and_misaligned_results_raise(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            retrieval_feature_service.extract(
                bm25_scores=[],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
            )

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[1.0],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
                bm25_ids=[],
                dense_ids=[],
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[float("nan")],
                dense_distances=[],
                bm25_docs=["document"],
                dense_docs=[],
                bm25_ids=["doc-id"],
                dense_ids=[],
            )

        with self.assertRaisesRegex(ValueError, "ids and .* equal lengths"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[1.0],
                dense_distances=[],
                bm25_docs=["document"],
                dense_docs=[],
                bm25_ids=[],
                dense_ids=[],
            )

        with self.assertRaisesRegex(ValueError, "dense_distances must contain only finite"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[],
                dense_distances=[float("inf")],
                bm25_docs=[],
                dense_docs=["document"],
                bm25_ids=[],
                dense_ids=["dense-id"],
            )

        with self.assertRaisesRegex(ValueError, "Missing required.*dense_ids"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
                bm25_ids=[],
            )

        with self.assertRaisesRegex(ValueError, "Missing required.*dense_ids"):
            retrieval_feature_service.extract(
                query="query",
                bm25_scores=[],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
                bm25_ids=[],
            )

        with self.assertRaisesRegex(ValueError, "Missing required.*query"):
            retrieval_feature_service.extract(
                bm25_scores=[],
                dense_distances=[],
                bm25_docs=[],
                dense_docs=[],
                bm25_ids=[],
                dense_ids=[],
            )

    def test_missing_predictor_features_raise(self):
        predictor = AdaptivePredictorV4.__new__(AdaptivePredictorV4)
        predictor.features = list(V4_RICH_FEATURES)
        predictor.model = StubModel()

        with self.assertRaisesRegex(ValueError, "Missing required"):
            predictor.predict(query="query", retrieval_features={})

    def test_predicted_weights_are_constrained_and_complementary(self):
        feature_map = {
            name: 0.25
            for name in V4_RICH_FEATURES
        }
        adaptive_predictor_v4.model = StubModel(-0.25)
        low = adaptive_predictor_v4.predict("query", feature_map)
        self.assertEqual(low["bm25_weight"], 0.0)
        self.assertEqual(low["dense_weight"], 1.0)

        adaptive_predictor_v4.model = StubModel(1.25)
        high = adaptive_predictor_v4.predict("query", feature_map)
        self.assertEqual(high["bm25_weight"], 1.0)
        self.assertEqual(high["dense_weight"], 0.0)
        self.assertAlmostEqual(
            high["dense_weight"],
            1.0 - high["bm25_weight"],
        )
        self.assertEqual(
            adaptive_predictor_v4.model.last_frame.shape,
            (1, len(V4_RICH_FEATURES)),
        )
        self.assertEqual(
            adaptive_predictor_v4.model.last_frame.dtype,
            np.float64,
        )

    def test_saved_model_artifact_accepts_the_named_17_feature_schema(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "adaptive_weight_model_v4_rich.pkl"
        )
        model = joblib.load(model_path)
        self.assertEqual(model.n_features_in_, len(V4_RICH_FEATURES))
        self.assertEqual(
            tuple(model.feature_names_in_),
            V4_RICH_FEATURES,
        )
        prediction = model.predict(np.zeros((1, len(V4_RICH_FEATURES))))
        self.assertEqual(np.asarray(prediction).shape, (1,))

    def test_fusion_uses_documented_weighted_sum(self):
        fusion = AdaptiveScoreFusion()
        results = fusion.fuse(
            dense_docs=["shared", "dense-only"],
            dense_distances=[0.1, 0.9],
            dense_ids=["shared", "dense-only"],
            dense_metadatas=[{"document_id": "shared"}, {"document_id": "dense-only"}],
            bm25_docs=["shared", "bm25-only"],
            bm25_scores=[10.0, 2.0],
            bm25_ids=["shared", "bm25-only"],
            bm25_metadatas=[{"document_id": "shared"}, {"document_id": "bm25-only"}],
            dense_weight=0.25,
            bm25_weight=0.75,
        )
        for item in results:
            expected = (
                0.25 * item["dense_component"]
                + 0.75 * item["bm25_component"]
            )
            self.assertAlmostEqual(item["fusion_score"], expected)


if __name__ == "__main__":
    unittest.main()
