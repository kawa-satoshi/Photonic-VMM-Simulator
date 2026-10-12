import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
import torch
from photonic_vmm import VMMConfig
from photonic_vmm.convert import convert
from photonic_vmm.data import load_rows, text_rows, validate_rows
from photonic_vmm.workflows import evaluate_batches, seed_everything, train_hat


class DataTests(unittest.TestCase):
    def test_source_selection(self):
        sources = ('tokens_file', 'text_file', 'arrow_file', 'dataset')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            load_rows()
        for pair in itertools.combinations(sources, 2):
            with self.subTest(sources=pair), self.assertRaisesRegex(ValueError, 'exactly one'):
                load_rows(**{name: 'unused' for name in pair})

    def test_local_text_loading(self):
        tokenizer = lambda text, add_special_tokens: {'input_ids': list(map(int, text.split()))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.txt'
            path.write_text('0 1 2\n3 4 5 6')
            rows, metadata = load_rows(text_file=path, tokenizer=tokenizer,
                                       sequence_length=3, max_rows=10)
            self.assertEqual(rows, [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(metadata['kind'], 'local_text')

    @unittest.skipUnless(importlib.util.find_spec('datasets'), 'Optional datasets is not installed')
    def test_local_arrow_loading_and_column_validation(self):
        from datasets import Dataset
        tokenizer = lambda text, add_special_tokens: {'input_ids': list(map(int, text.split()))}
        with tempfile.TemporaryDirectory() as directory:
            Dataset.from_dict({'body': ['0 1 2 3 4', '5 6 7 8'], 'number': [1, 2]}).save_to_disk(directory)
            path = next(Path(directory).glob('*.arrow'))
            rows, metadata = load_rows(arrow_file=path, text_column='body', tokenizer=tokenizer,
                                       sequence_length=2, max_rows=3)
            self.assertEqual(rows, [[0, 1], [2, 3], [5, 6]])
            self.assertEqual(metadata['kind'], 'local_arrow')
            self.assertEqual(metadata['text_column'], 'body')
            self.assertEqual(metadata['rows'], 3)
            with self.assertRaisesRegex(ValueError, 'Missing dataset text column'):
                load_rows(arrow_file=path, tokenizer=tokenizer)
            with self.assertRaisesRegex(ValueError, 'must contain strings'):
                load_rows(arrow_file=path, text_column='number', tokenizer=tokenizer)

    def test_document_boundaries_and_short_tails(self):
        tokenizer = lambda text, add_special_tokens: {'input_ids': list(map(int, text.split()))}
        rows = text_rows(['0 1 2 3 4', '5 6'], tokenizer, 2, 10)
        self.assertEqual(rows, [[0, 1], [2, 3], [5, 6]])

    def test_tokens_validate_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.json'
            path.write_text(json.dumps([[0, 1], [2, 3, 4]]))
            rows, metadata = load_rows(tokens_file=path, max_rows=1)
            self.assertEqual(rows, [[0, 1]])
            self.assertEqual(metadata['rows'], 1)
            self.assertEqual(len(metadata['rows_sha256']), 64)
            with self.assertRaises(ValueError):
                load_rows(tokens_file=path, text_file=path)

    def test_invalid_rows(self):
        for rows in ([], [[1]], [[True, 2]], [[-1, 0]], [['1', 2]], {'tokens': [1, 2]}):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_rows(rows)


class WorkflowTests(unittest.TestCase):
    def test_training_improves_and_freezes_unselected_parameters(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 8)
                self.head = torch.nn.Linear(8, 4)

            def forward(self, input_ids, use_cache=False):
                return types.SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        seed_everything(7)
        model = Toy()
        config = VMMConfig(tile_in=4, tile_out=4, group_size=4,
                           constant_range_mode='zero', output_tile_execution='temporal_multiplexed',
                           x_bits=8, w_bits=8, y_bits=8,
                           x_noise=True, w_noise=True, y_noise=True,
                           x_std=.001, w_std=.001, y_std=.001)
        convert(model, config, ['head'])
        rows = [[0, 1, 2, 3, 0, 1, 2, 3]]
        original = model.embedding.weight.detach().clone()
        before = evaluate_batches(model, rows, 'cpu')['loss']
        report = train_hat(model, rows, 'cpu', ['head'], steps=12, learning_rate=.05)
        after = evaluate_batches(model, rows, 'cpu')['loss']
        self.assertLess(after, before * .6)
        self.assertTrue(all(norm > 0 for norm in report['gradient_norms']))
        torch.testing.assert_close(original, model.embedding.weight, rtol=0, atol=0)

    def test_invalid_training_settings(self):
        for steps, rate in ((0, .1), (1, 0), (1, float('nan'))):
            with self.assertRaises(ValueError):
                train_hat(torch.nn.Linear(2, 2), [[0, 1]], 'cpu', [],
                          steps=steps, learning_rate=rate)


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional transformers is not installed')
class OtherArchitectureTests(unittest.TestCase):
    def test_random_llama_conversion_and_tied_parameters(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        seed_everything(13)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, tie_word_embeddings=True,
            attention_dropout=0.)).float().eval()
        tokens = torch.tensor([[1, 3, 5, 7]])
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits
        targets = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
        config = VMMConfig(tile_in=8, tile_out=8, group_size=8,
                           constant_range_mode='zero', output_tile_execution='temporal_multiplexed')
        convert(model, config)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        with torch.no_grad():
            actual = model(tokens, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
