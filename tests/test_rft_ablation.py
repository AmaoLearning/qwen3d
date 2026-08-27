import importlib.util
import tempfile
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RFTAblationTest(unittest.TestCase):
    def test_campaign_covers_control_and_requested_variants(self):
        module = load_module("rft_ablation", "repro/run_rft_ablation.py")
        names = [name for name, _, _ in module.VARIANTS]
        self.assertEqual(
            names,
            [
                "original_ce",
                "paper_ratio",
                "neg_log_phi",
                "focal_g0p5",
                "focal_g1",
                "focal_g2",
                "one_minus_log1p",
            ],
        )

    def test_eval_log_parser_keeps_both_qa_datasets(self):
        module = load_module("rft_ablation_parser", "repro/run_rft_ablation.py")
        text = """\
Evaluating on sqa3d_ref_scannet_test_single_batched, idx: 0
exact match: 0.25,
cider: 0.1, bleu: 0.2, meteor: 0.3, rouge: 0.4,
Evaluating on scanqa_ref_scannet_test_single_batched, idx: 1
exact match: 0.5,
cider: 0.6, bleu: 0.7, meteor: 0.8, rouge: 0.9,
"""
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "eval.log"
            log_path.write_text(text)
            metrics = module.parse_eval_metrics(log_path)
        self.assertEqual(metrics["sqa3d"]["exact_match"], 0.25)
        self.assertEqual(metrics["sqa3d"]["rouge"], 0.4)
        self.assertEqual(metrics["scanqa"]["exact_match"], 0.5)
        self.assertEqual(metrics["scanqa"]["meteor"], 0.8)


if __name__ == "__main__":
    unittest.main()
