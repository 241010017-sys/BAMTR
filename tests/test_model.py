"""Model invariants: masking, inference isolation and autoregressive feedback."""
from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from bamtr.model import BAMTR


def example_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    batch = {
        "obs_coords": torch.rand(2, 5, 2, generator=generator),
        "obs_time": torch.linspace(0, 1, 5).reshape(1, 5, 1).repeat(2, 1, 1),
        "obs_stay": torch.randint(0, 2, (2, 5, 1), generator=generator).float(),
        "obs_lengths": torch.tensor([5, 3]),
        "hist_coords": torch.rand(2, 3, 4, 2, generator=generator),
        "hist_time": torch.rand(2, 3, 4, 1, generator=generator),
        "hist_stay": torch.randint(0, 2, (2, 3, 4, 1), generator=generator).float(),
        "hist_lengths": torch.tensor([[4, 2, 0], [3, 0, 0]]),
        "hist_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "target_time": torch.tensor([[[0.2], [0.5], [0.7], [0.9]], [[0.4], [0.8], [0.0], [0.0]]]),
        "target_coords": torch.rand(2, 4, 2, generator=generator),
        "target_stay": torch.randint(0, 2, (2, 4), generator=generator).float(),
        "target_mask": torch.tensor([[True, True, True, True], [True, True, False, False]]),
        "environment": torch.rand(2, 5, generator=generator),
    }
    return batch


def small_model(**overrides) -> BAMTR:
    config = {"encoder_hidden_size": 5, "decoder_hidden_size": 7,
              "num_layers": 1, "dropout": 0.0, "semantic_embedding_dim": 3,
              "environment_embedding_dim": 4, "coord_head_dim": 9,
              **overrides}
    return BAMTR(config, environment_dim=5)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.batch = example_batch()

    def test_variable_lengths_and_backward(self):
        model = small_model()
        output = model(self.batch, teacher_forcing_ratio=0.5)
        self.assertEqual(output["coords"].shape, (2, 4, 2))
        self.assertEqual(output["stay_logits"].shape, (2, 4))
        self.assertEqual(output["history_attention"].shape, (2, 3))
        mask = self.batch["target_mask"]
        self.assertTrue(torch.all(output["coords"][~mask] == 0))
        self.assertTrue(torch.all(output["stay_logits"][~mask] == 0))
        torch.testing.assert_close(output["history_attention"].sum(-1), torch.ones(2))
        self.assertTrue(torch.all(output["history_attention"][~self.batch["hist_mask"]] == 0))
        loss = F.mse_loss(output["coords"][mask], self.batch["target_coords"][mask])
        loss = loss + F.binary_cross_entropy_with_logits(output["stay_logits"][mask], self.batch["target_stay"][mask])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(gradients), 10)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_padding_values_never_change_predictions(self):
        model = small_model().eval()
        original = model(self.batch)
        changed = {key: value.clone() for key, value in self.batch.items()}
        for name in ("obs_coords", "obs_time", "obs_stay"):
            changed[name][1, 3:] = float("nan")
        for name in ("hist_coords", "hist_time", "hist_stay"):
            for b in range(2):
                for k in range(3):
                    changed[name][b, k, int(changed["hist_lengths"][b, k]):] = float("nan")
        # Even a nonzero length must not activate a history whose mask is false.
        changed["hist_lengths"][0, 2] = 4
        changed["target_time"][1, 2:] = float("nan")
        changed["target_coords"][1, 2:] = float("nan")
        result = model(changed)
        for name in original:
            torch.testing.assert_close(result[name], original[name])

    def test_packed_batch_matches_individual_sequences(self):
        model = small_model().eval()
        together = model(self.batch)
        for b, count in enumerate([4, 2]):
            single = {key: value[b:b + 1].clone() for key, value in self.batch.items()}
            result = model(single)
            torch.testing.assert_close(result["coords"][0, :count], together["coords"][b, :count], atol=1e-6, rtol=1e-5)

    def test_empty_history_is_finite_and_equivalent_to_absent_slots(self):
        model = small_model().eval()
        self.batch["hist_mask"][:] = False
        self.batch["hist_lengths"][:] = 0
        for name in ("hist_coords", "hist_time", "hist_stay"):
            self.batch[name][:] = float("nan")
        padded = model(self.batch)
        self.assertTrue(torch.isfinite(padded["coords"]).all())
        self.assertTrue(torch.all(padded["history_attention"] == 0))
        empty = {key: (value[:, :0] if key.startswith("hist_") else value)
                 for key, value in self.batch.items()}
        output = model(empty)
        self.assertEqual(output["history_attention"].shape, (2, 0))
        torch.testing.assert_close(output["coords"], padded["coords"])

    def test_evaluation_never_reads_supervision_even_with_forcing_requested(self):
        model = small_model().eval()
        reference = model(self.batch, teacher_forcing_ratio=1.0)
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["target_coords"][:] = float("nan")
        changed["target_stay"][:] = float("nan")
        result = model(changed, teacher_forcing_ratio=1.0)
        torch.testing.assert_close(result["coords"], reference["coords"])
        del changed["target_coords"]
        del changed["target_stay"]
        torch.testing.assert_close(model(changed)["coords"], reference["coords"])

    def test_coordinate_forcing_is_training_only_and_never_uses_stay_labels(self):
        model = small_model().train()
        reference = model(self.batch, teacher_forcing_ratio=1.0)
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["target_stay"][:] = float("nan")
        torch.testing.assert_close(model(changed, 1.0)["coords"], reference["coords"])
        changed["target_coords"] = changed["target_coords"] + 10
        result = model(changed, 1.0)
        torch.testing.assert_close(result["coords"][:, 0], reference["coords"][:, 0])
        self.assertFalse(torch.allclose(result["coords"][:, 1], reference["coords"][:, 1]))
        torch.testing.assert_close(model(changed, 0.0)["coords"], model(self.batch, 0.0)["coords"])

    def test_dropout_is_active_for_a_single_gru_layer(self):
        model = small_model(dropout=0.5).train()
        first = model(self.batch)["coords"]
        second = model(self.batch)["coords"]
        self.assertFalse(torch.allclose(first, second))
        model.eval()
        torch.testing.assert_close(model(self.batch)["coords"], model(self.batch)["coords"])

    def test_query_time_extension_is_explicit(self):
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["target_time"] = changed["target_time"] + 0.05
        faithful = small_model().eval()
        torch.testing.assert_close(faithful(self.batch)["coords"], faithful(changed)["coords"])
        extended = small_model(query_time_conditioning=True).eval()
        self.assertFalse(torch.allclose(extended(self.batch)["coords"], extended(changed)["coords"]))

    def test_optional_modules_and_multilayer_projection(self):
        for flag in ("use_history", "use_context_adaptor", "use_mobility", "use_environment", "use_local_attention"):
            with self.subTest(flag=flag):
                output = small_model(**{flag: False}).eval()(self.batch)
                self.assertTrue(torch.isfinite(output["coords"]).all())
        output = small_model(num_layers=2).eval()(self.batch)
        self.assertEqual(output["coords"].shape, (2, 4, 2))

    def test_empty_targets_and_invalid_observation_lengths(self):
        model = small_model().eval()
        empty = {key: (value[:, :0] if key.startswith("target_") else value)
                 for key, value in self.batch.items()}
        self.assertEqual(model(empty)["coords"].shape, (2, 0, 2))
        self.batch["obs_lengths"][0] = 0
        with self.assertRaises(ValueError):
            model(self.batch)


if __name__ == "__main__":
    unittest.main()
