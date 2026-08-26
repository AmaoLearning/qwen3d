import math
import importlib.util
import unittest
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen3d/modeling/rft_loss.py"
_SPEC = importlib.util.spec_from_file_location("qwen3d_rft_loss", _MODULE_PATH)
_RFT_LOSS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_RFT_LOSS)
compute_rft_loss = _RFT_LOSS.compute_rft_loss
combine_post_training_losses = _RFT_LOSS.combine_post_training_losses


def logits_for_target_probabilities(probabilities, targets):
    rows = []
    for probability, target in zip(probabilities, targets):
        other = 1 - probability
        values = [math.log(other), math.log(other)]
        values[target] = math.log(probability)
        rows.append(values)
    # Qwen-3D generation mode returns one trailing, unsupervised logit.
    rows.append([0.0, 0.0])
    return torch.tensor([rows], dtype=torch.float64, requires_grad=True)


class RFTLossTest(unittest.TestCase):
    def setUp(self):
        self.targets = [0, 1]
        self.labels = torch.tensor([[-100, -100, *self.targets]])
        self.answer_start = torch.tensor([2])
        self.full_logits = logits_for_target_probabilities(
            [0.8, 0.6], self.targets
        )
        self.phi_logits = logits_for_target_probabilities(
            [0.25, 0.5], self.targets
        )
        self.theta_logits = logits_for_target_probabilities(
            [0.5, 0.25], self.targets
        )

    def compute(self, loss_type, gamma=1.0):
        return compute_rft_loss(
            full_logits=self.full_logits,
            labels=self.labels,
            answer_start=self.answer_start,
            loss_type=loss_type,
            phi_logits=self.phi_logits,
            theta_blind_logits=self.theta_logits,
            gamma=gamma,
        )

    def test_original_matches_mean_cross_entropy(self):
        result = compute_rft_loss(
            full_logits=self.full_logits,
            labels=self.labels,
            answer_start=self.answer_start,
            loss_type="original",
        )
        expected = (-math.log(0.8) - math.log(0.6)) / 2
        self.assertAlmostEqual(result.loss.item(), expected, places=6)

    def test_paper_ratio(self):
        result = self.compute("paper_ratio")
        full = torch.tensor([-math.log(0.8), -math.log(0.6)])
        phi = torch.tensor([-math.log(0.25), -math.log(0.5)])
        theta = torch.tensor([-math.log(0.5), -math.log(0.25)])
        expected = (full * (phi / theta)).mean()
        self.assertAlmostEqual(result.loss.item(), expected.item(), places=6)

    def test_negative_log_phi(self):
        result = self.compute("neg_log_phi")
        full = torch.tensor([-math.log(0.8), -math.log(0.6)])
        weight = torch.tensor([-math.log(0.25), -math.log(0.5)])
        self.assertAlmostEqual(result.loss.item(), (full * weight).mean().item(), 6)

    def test_focal_gamma_half_one_two(self):
        losses = []
        for gamma in (0.5, 1.0, 2.0):
            result = self.compute("focal", gamma=gamma)
            full = torch.tensor([-math.log(0.8), -math.log(0.6)])
            weight = torch.tensor([(1 - 0.25) ** gamma, (1 - 0.5) ** gamma])
            self.assertAlmostEqual(
                result.loss.item(), (full * weight).mean().item(), places=6
            )
            losses.append(result.loss.item())
        self.assertGreater(losses[0], losses[1])
        self.assertGreater(losses[1], losses[2])

    def test_one_minus_log1p(self):
        result = self.compute("one_minus_log1p")
        full = torch.tensor([-math.log(0.8), -math.log(0.6)])
        probability = torch.tensor([0.25, 0.5])
        expected = (full * (1 - torch.log1p(probability))).mean()
        self.assertAlmostEqual(result.loss.item(), expected.item(), places=6)

    def test_reference_weights_are_detached(self):
        result = self.compute("paper_ratio")
        result.loss.backward()
        self.assertIsNotNone(self.full_logits.grad)
        self.assertIsNone(self.phi_logits.grad)
        self.assertIsNone(self.theta_logits.grad)

    def test_original_and_rft_coefficients_are_independent(self):
        original = torch.tensor(2.0)
        rft = torch.tensor(3.0)
        combined = combine_post_training_losses(
            original,
            rft,
            original_coef=0.25,
            rft_coef=1.5,
        )
        self.assertAlmostEqual(combined.item(), 5.0)

    def test_invalid_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown RFT loss type"):
            self.compute("not-a-loss")


if __name__ == "__main__":
    unittest.main()
