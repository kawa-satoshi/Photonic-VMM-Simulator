"""Offline end-to-end held-out HAT, safe checkpoint reload, and paired noise."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional transformers is not installed')
class CLITests(unittest.TestCase):
    def test_heldout_checkpoint_replays_same_evaluation(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=24, hidden_size=16, intermediate_size=24,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32, tie_word_embeddings=True,
                attention_dropout=0.)).float()
            model.save_pretrained(root / 'model')
            (root / 'train.json').write_text(json.dumps([[1, 2, 3, 4]]))
            (root / 'heldout.json').write_text(json.dumps([[5, 6, 7, 8]]))
            (root / 'targets.json').write_text(json.dumps(['lm_head']))
            (root / 'config.json').write_text(json.dumps(dict(
                constant_range_mode='zero', tile_in=8, tile_out=8, group_size=8,
                x_bits=8, w_bits=8, y_bits=8, x_noise=True, w_noise=True, y_noise=True)))
            cli = Path(__file__).resolve().parents[1] / 'evaluate.py'
            common = [sys.executable, str(cli), '--model', str(root / 'model'),
                      '--config', str(root / 'config.json'), '--targets-file', str(root / 'targets.json'),
                      '--protect-threshold', '6', '--protect-aggregation', 'batch_sequence', '--seed', '42']
            env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', USE_TF='0', OMP_NUM_THREADS='1')
            first = subprocess.run(common + [
                '--tokens-file', str(root / 'train.json'), '--validation-tokens-file', str(root / 'heldout.json'),
                '--hat-steps', '2', '--learning-rate', '.001', '--checkpoint', str(root / 'hat.pt'),
                '--output', str(root / 'first.json')], env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(common + [
                '--tokens-file', str(root / 'heldout.json'), '--load-checkpoint', str(root / 'hat.pt'),
                '--output', str(root / 'second.json')], env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            a = json.loads((root / 'first.json').read_text())
            b = json.loads((root / 'second.json').read_text())
            self.assertEqual(a['evaluation_data']['rows_sha256'], b['data']['rows_sha256'])
            self.assertNotEqual(a['data']['rows_sha256'], a['evaluation_data']['rows_sha256'])
            self.assertEqual(a['evaluation_noise_index'], b['evaluation_noise_index'])
            self.assertEqual(a['metrics'], b['metrics'])
            self.assertTrue(b['checkpoint_loaded'])
            self.assertEqual(a['training']['steps'], 2)

            overlap = subprocess.run(common + [
                '--tokens-file', str(root / 'train.json'), '--validation-tokens-file', str(root / 'train.json'),
                '--hat-steps', '1', '--output', str(root / 'invalid.json')],
                env=env, capture_output=True, text=True)
            self.assertNotEqual(overlap.returncode, 0)
            self.assertIn('identical token row', overlap.stderr)
            self.assertFalse((root / 'invalid.json').exists())
