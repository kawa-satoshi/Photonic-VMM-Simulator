"""Independent scalar Q/DQ and multiply-add reference, plus integration contracts."""
import dataclasses
import itertools
import math
import unittest
from unittest.mock import patch
import torch
from photonic_vmm.core import VMMConfig, PhotonicLinear, simulate_linear, quantize, draw_noise
from photonic_vmm.convert import convert


def cfg(**kwargs):
    return VMMConfig(**dict(constant_range_mode='zero', output_tile_execution='spatial_parallel',
                           tile_in=3, tile_out=3, group_size=3, **kwargs))


def scalar_q(row, bits):
    # Independent formula uses the quantization step, the reciprocal of core scale.
    if bits is None or max(row) == min(row):
        return row[:]
    step = (max(row)-min(row))/(2**bits-1)
    zero = -min(row)/step-2**(bits-1)
    return [(round(v/step+zero)-zero)*step for v in row]


def prescribed(stage, idx):
    return (idx % 3 - 1) * {'x':.031, 'w':.023, 'y':.079}[stage]


def injected_noise(t, config, stage, *_):
    values = [prescribed(stage, i) if getattr(config, stage+'_noise') else 0. for i in range(t.numel())]
    return torch.tensor(values, dtype=t.dtype).reshape_as(t)


def scalar_reference(x, w, bias, c):
    # Deliberately no torch matmul, production quantizer, or production noise calls.
    rows=x.reshape(-1,x.shape[-1]).tolist(); weights=w.tolist()
    result=[[0. for _ in weights] for _ in rows]
    for k in range(0,len(weights[0]),c.tile_in):
        klen=min(c.tile_in,len(weights[0])-k)
        xx=[scalar_q(row[k:k+klen],c.x_bits) for row in rows]
        if c.x_noise:
            xx=[[v+prescribed('x',m*klen+j) for j,v in enumerate(row)] for m,row in enumerate(xx)]
        for o in range(0,len(weights),c.tile_out):
            olen=min(c.tile_out,len(weights)-o)
            ww=[scalar_q(row[k:k+klen],c.w_bits) for row in weights[o:o+olen]]
            if c.w_noise:
                ww=[[v+prescribed('w',n*klen+j) for j,v in enumerate(row)] for n,row in enumerate(ww)]
            for m,row in enumerate(xx):
                partial=[sum(a*b for a,b in zip(row,wr)) for wr in ww]
                if c.y_noise:
                    partial=[v+prescribed('y',m*olen+n) for n,v in enumerate(partial)]
                adc=scalar_q(partial,c.y_bits)
                for n,v in enumerate(adc):result[m][o+n]+=v
    if bias is not None:
        result=[[v+bias[n].item() for n,v in enumerate(row)] for row in result]
    return torch.tensor(result,dtype=torch.float32).reshape(*x.shape[:-1],len(weights))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        g=torch.Generator().manual_seed(19)
        self.x=torch.randn(2,3,7,generator=g)
        self.w=torch.randn(5,7,generator=g)
        self.b=torch.randn(5,generator=g)

    def test_disabled_and_tail_tiles(self):
        for x in (self.x,self.x.transpose(0,1),self.x[0,0]):
            for bias in (None,self.b):
                y=simulate_linear(x,self.w,bias,cfg())
                torch.testing.assert_close(y,torch.nn.functional.linear(x,self.w,bias),rtol=1e-5,atol=2e-6)

    def test_independent_scalar_reference_all_switches(self):
        for flags in itertools.product((False,True),repeat=3):
            c=cfg(x_bits=3,w_bits=3,y_bits=3,x_noise=flags[0],w_noise=flags[1],y_noise=flags[2])
            with self.subTest(flags=flags),patch('photonic_vmm.core.draw_noise',side_effect=injected_noise):
                actual=simulate_linear(self.x,self.w,self.b,c)
                expected=scalar_reference(self.x,self.w,self.b,c)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)

    def test_confirmed_defaults_and_default_head_scope(self):
        c=VMMConfig()
        self.assertEqual(c.constant_range_mode,'legacy_abs')
        self.assertEqual(c.output_tile_execution,'spatial_parallel')
        c=dataclasses.replace(c,x_noise=True)
        constant=torch.full((2,3),-2.)
        self.assertGreater(draw_noise(constant,c,'x','layer',0,0,0).abs().max().item(),0)
        self.assertTrue(torch.equal(draw_noise(constant*0,c,'x','layer',0,0,0),constant*0))
        m=torch.nn.Module();m.proj=torch.nn.Linear(3,3);m.lm_head=torch.nn.Linear(3,5)
        manifest=convert(m,c)
        self.assertEqual([r['path'] for r in manifest],['proj','lm_head'])
        m=torch.nn.Module();m.proj=torch.nn.Linear(3,3);m.lm_head=torch.nn.Linear(3,5)
        manifest=convert(m,c,exclude_paths=['lm_head'])
        self.assertEqual([r['path'] for r in manifest],['proj'])
        self.assertIsInstance(m.lm_head,torch.nn.Linear)

    def test_aliases_and_duplicate_exclusions_rejected_before_mutation(self):
        m=torch.nn.Module();m.a=torch.nn.Linear(3,3);m.b=m.a
        with self.assertRaisesRegex(ValueError,'Aliased'):
            convert(m,VMMConfig())
        self.assertIs(m.a,m.b);self.assertIsInstance(m.a,torch.nn.Linear)
        m=torch.nn.Module();m.a=torch.nn.Linear(3,3);m.b=torch.nn.Linear(3,3)
        with self.assertRaisesRegex(ValueError,'Duplicate target'):
            convert(m,VMMConfig(),['a','a','b'],exclude_paths=['a'])
        with self.assertRaisesRegex(ValueError,'Duplicate excluded'):
            convert(m,VMMConfig(),exclude_paths=['a','a'])
        self.assertIsInstance(m.a,torch.nn.Linear)

    def test_bias_once_and_no_final_quantizer(self):
        c=cfg(x_bits=3,w_bits=3,y_bits=2)
        y=simulate_linear(self.x,self.w,None,c)
        torch.testing.assert_close(simulate_linear(self.x,self.w,self.b,c),y+self.b,rtol=0,atol=0)
        self.assertFalse(torch.allclose(y[...,:3],quantize(y[...,:3],2)))

    def test_legacy_whole_output_differs(self):
        c=cfg(x_bits=3,w_bits=3,y_bits=3)
        def group_q(t):return torch.cat([quantize(t[...,k:k+3],3) for k in range(0,t.shape[-1],3)],-1)
        old=group_q(torch.nn.functional.linear(group_q(self.x),group_q(self.w)))+self.b
        tiled=simulate_linear(self.x,self.w,self.b,c)
        self.assertGreater((old-tiled).abs().max().item(),.01)

    def test_seed_replay_and_per_forward_weight(self):
        c=cfg(w_noise=True)
        source=torch.nn.Linear(7,5);source.weight=torch.nn.Parameter(self.w);source.bias=torch.nn.Parameter(self.b)
        m=PhotonicLinear(source,c,'test.layer')
        a=m(self.x);b=m(self.x);m.reset_noise();d=m(self.x)
        self.assertTrue(torch.equal(a,d));self.assertFalse(torch.equal(a,b))
        # Same W errors for repeated tokens in a forward.
        y=m(self.x[0,0].repeat(4,1))
        self.assertTrue(torch.equal(y[0],y[3]))

    def test_stage_rng_independence(self):
        c=cfg(w_noise=True)
        n=draw_noise(self.w,c,'w','layer',3,0,0)
        d=dataclasses.replace(c,x_noise=True,y_noise=True)
        self.assertTrue(torch.equal(n,draw_noise(self.w,d,'w','layer',3,0,0)))
        torch.manual_seed(123);state=torch.get_rng_state()
        simulate_linear(self.x,self.w,self.b,d)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_real_noise_range_axes_and_std(self):
        c=cfg(x_noise=True,w_noise=True,y_noise=True,x_std=.1,w_std=.1,y_std=.1)
        t=torch.tensor([[0.,2.,1.],[10.,14.,12.]])
        def ones(shape,**kwargs):return torch.ones(shape,dtype=kwargs['dtype'],device=kwargs['device'])
        with patch('torch.randn',side_effect=ones):
            for stage in ('x','w','y'):
                torch.testing.assert_close(draw_noise(t,c,stage,'l',0,0,0),torch.tensor([[.2]*3,[.4]*3]))

    def test_fp32_sequential_accumulation(self):
        c=dataclasses.replace(cfg(),tile_in=1,tile_out=1,group_size=1)
        y=simulate_linear(torch.ones(1,3),torch.tensor([[2**24,1.,-2**24]]),None,c)
        self.assertEqual(y.item(),0.)  # double accumulation then cast would be 1

    def test_constant_range_modes_and_zero(self):
        c=cfg(w_noise=True)
        t=torch.full((4,3),2.)
        self.assertTrue(torch.equal(draw_noise(t,c,'w','l',0,0,0),torch.zeros_like(t)))
        legacy=dataclasses.replace(c,constant_range_mode='legacy_abs')
        self.assertGreater(draw_noise(t,legacy,'w','l',0,0,0).abs().max().item(),0)
        self.assertEqual(draw_noise(t*0,legacy,'w','l',0,0,0).abs().max().item(),0)
        self.assertTrue(torch.equal(quantize(t,2),t))

    def test_x_scope(self):
        c=cfg(x_noise=True)
        self.assertTrue(torch.equal(draw_noise(self.x,c,'x','l',0,0,0),draw_noise(self.x,c,'x','l',0,0,3)))
        c=dataclasses.replace(c,output_tile_execution='temporal_multiplexed')
        self.assertFalse(torch.equal(draw_noise(self.x,c,'x','l',0,0,0),draw_noise(self.x,c,'x','l',0,0,3)))

    def test_round_ties_and_short_constant_adc(self):
        t=torch.tensor([[0.,1/6,0.5,5/6,1.]])
        torch.testing.assert_close(quantize(t,2),torch.tensor([[0.,0.,2/3,2/3,1.]]))
        single=torch.tensor([[2.],[7.]])
        self.assertTrue(torch.equal(quantize(single,2),single))

    def test_trace_tiles(self):
        seen=[]
        y=simulate_linear(self.x,self.w,self.b,cfg(),trace=lambda k,o,p,q:seen.append((k,o,tuple(p.shape),p.dtype)))
        self.assertEqual(len(seen),6)
        self.assertEqual(seen[-1],(6,3,(2,3,2),torch.float32))
        self.assertEqual(y.dtype,torch.float32)

    def test_tiny_range_and_large_offset_are_finite(self):
        for t in (torch.tensor([[0.,1e-39,2e-39]]),torch.tensor([[1e8,1e8+8,1e8+16]])):
            q=quantize(t,8)
            self.assertTrue(torch.isfinite(q).all())
            self.assertEqual(q[0,0].item(),t[0,0].item())
            self.assertEqual(q[0,-1].item(),t[0,-1].item())

    def test_invalid_config_and_input(self):
        for change in ({'group_size':4},{'tile_in':0},{'x_std':-1},{'w_std':math.nan},{'y_bits':1},{'weight_noise_mode':'fixed'}):
            with self.subTest(change=change),self.assertRaises(ValueError):dataclasses.replace(cfg(),**change)
        with self.assertRaises(ValueError):simulate_linear(self.x.double(),self.w,None,cfg())
        with self.assertRaises(ValueError):simulate_linear(self.x,self.w,torch.ones(4),cfg())

    def test_nonfinite_and_shape_validation(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            for index in range(3):
                tensors=[self.x.clone(),self.w.clone(),self.b.clone()]
                tensors[index].reshape(-1)[0]=value
                with self.subTest(value=value,index=index),self.assertRaises(ValueError):
                    simulate_linear(*tensors,cfg())
        with self.assertRaises(ValueError):simulate_linear(self.x,self.w[:,:6],None,cfg())
        with self.assertRaises(ValueError):simulate_linear(self.x,torch.zeros(0,7),None,cfg())
        with self.assertRaises(ValueError):simulate_linear(self.x,self.w,None,cfg(),forward_index=-1)
        y=simulate_linear(torch.zeros(0,2,7),self.w,None,cfg())
        self.assertEqual(y.shape,(0,2,5))

    def test_tied_parameter_and_source_unchanged(self):
        m=torch.nn.Module();m.embed=torch.nn.Embedding(5,7);m.head=torch.nn.Linear(7,5,bias=False)
        m.head.weight=m.embed.weight;m.eval();before=m.embed.weight.detach().clone()
        convert(m,cfg(x_bits=3,w_bits=3,y_bits=3,w_noise=True),['head'])
        m.head(self.x)
        self.assertIs(m.head.weight,m.embed.weight)
        self.assertTrue(torch.equal(before,m.embed.weight))
        self.assertFalse(m.head.training)

    def test_layer_noise_keys_are_distinct(self):
        c=cfg(w_noise=True)
        a=draw_noise(self.w,c,'w','layer.0',0,0,0)
        b=draw_noise(self.w,c,'w','layer.1',0,0,0)
        self.assertFalse(torch.equal(a,b))

    def test_converter_exact_path_and_atomic_validation(self):
        m=torch.nn.Module();m.q_proj=torch.nn.Linear(7,5);m.not_q_proj_aux=torch.nn.Linear(7,5)
        weight=m.q_proj.weight
        with self.assertRaises(AttributeError):convert(m,cfg(),['q_proj','missing'])
        self.assertIsInstance(m.q_proj,torch.nn.Linear)
        manifest=convert(m,cfg(),['q_proj'])
        self.assertIsInstance(m.q_proj,PhotonicLinear)
        self.assertIsInstance(m.not_q_proj_aux,torch.nn.Linear)
        self.assertIs(m.q_proj.weight,weight)
        self.assertEqual(manifest[0]['path'],'q_proj')


if __name__=='__main__':unittest.main()
