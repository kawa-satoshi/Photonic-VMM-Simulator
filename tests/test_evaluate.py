import types
import math
import unittest
import torch
from photonic_vmm.workflows import evaluate_batches


class EvaluationTests(unittest.TestCase):
    def test_shift_and_token_weighting(self):
        class Toy(torch.nn.Module):
            def forward(self,input_ids,use_cache):
                self.assert_cache=use_cache
                return types.SimpleNamespace(logits=torch.zeros(*input_ids.shape,4))
        model=Toy()
        r=evaluate_batches(model,[[0,1,2],[3,2,1,0]],'cpu')
        self.assertEqual(r['predicted_tokens'],5)
        self.assertAlmostEqual(r['perplexity'],4.,places=5)
        self.assertFalse(model.assert_cache)

    def test_no_tokens(self):
        with self.assertRaises(ValueError):evaluate_batches(torch.nn.Module(),[], 'cpu')

    def test_nonuniform_shift_and_weighting(self):
        class Toy(torch.nn.Module):
            def forward(self,input_ids,use_cache):
                length=input_ids.shape[1]
                return types.SimpleNamespace(logits=torch.tensor([[[0.,float(i+1),-1.] for i in range(length)]]))
        rows=[[0,1],[2,0,2,1]]
        result=evaluate_batches(Toy(),rows,'cpu')
        nll=[]
        for row in rows:
            for i,target in enumerate(row[1:]):
                logits=[0.,float(i+1),-1.]
                nll.append(math.log(sum(math.exp(v) for v in logits))-logits[target])
        self.assertEqual(result['predicted_tokens'],4)
        self.assertAlmostEqual(result['loss'],sum(nll)/4,places=6)
