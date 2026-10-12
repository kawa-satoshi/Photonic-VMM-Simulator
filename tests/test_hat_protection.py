"""Independent forward/gradient references and an actual short optimization run."""
import dataclasses
import unittest
from unittest.mock import patch
import torch
from photonic_vmm.core import simulate_linear, PhotonicLinear, transform
from photonic_vmm.protection import MaxAbsPolicy
from test_core import cfg, scalar_q, prescribed, injected_noise


class HATProtectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.x = torch.randn(2, 3, 7)
        self.w = torch.randn(5, 7)
        self.b = torch.randn(5)

    def test_exact_forward_train_eval_and_identity_transform(self):
        c = cfg(x_bits=3,w_bits=3,y_bits=3,x_noise=True,w_noise=True,y_noise=True)
        x = self.x.clone().requires_grad_()
        q = transform(x,3,c,'x','l',0,0,0)
        upstream = torch.randn_like(x)
        (q*upstream).sum().backward()
        torch.testing.assert_close(x.grad,upstream,rtol=0,atol=0)
        a = simulate_linear(x,self.w,self.b,c)
        with torch.no_grad(): b = simulate_linear(x,self.w,self.b,c)
        self.assertTrue(torch.equal(a,b))

    def test_independent_tiled_gradient_reference(self):
        # Identity STE gives dX = upstream @ noisy quantized W, not original W.
        c = cfg(x_bits=3,w_bits=3,y_bits=3,x_noise=True,w_noise=True,y_noise=True)
        mask = torch.tensor([False,True,False,False,True,False,False])
        x = self.x.clone().requires_grad_(); w = self.w.clone().requires_grad_(); b=self.b.clone().requires_grad_()
        upstream=torch.randn(2,3,5)
        with patch('photonic_vmm.core.draw_noise',side_effect=injected_noise):
            y=simulate_linear(x,w,b,c,protection=lambda _:mask)
        (y*upstream).sum().backward()
        xx=self.x.reshape(-1,7).tolist(); ww=self.w.tolist(); gg=upstream.reshape(-1,5).tolist()
        dx=[[0.]*7 for _ in xx]; dw=[[0.]*7 for _ in ww]
        for k in range(0,7,3):
            kl=min(3,7-k)
            xq=[scalar_q([0. if mask[k+j] else row[k+j] for j in range(kl)],3) for row in xx]
            for o in range(0,5,3):
                ol=min(3,5-o)
                wq=[scalar_q(row[k:k+kl],3) for row in ww[o:o+ol]]
                for m,row in enumerate(xx):
                    for n in range(ol):
                        for j in range(kl):
                            if mask[k+j]:
                                a=row[k+j]; v=ww[o+n][k+j]
                            else:
                                a=xq[m][j]+prescribed('x',m*kl+j)
                                v=wq[n][j]+prescribed('w',n*kl+j)
                            dx[m][k+j]+=gg[m][o+n]*v
                            dw[o+n][k+j]+=gg[m][o+n]*a
        torch.testing.assert_close(x.grad,torch.tensor(dx).reshape_as(x),atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(w.grad,torch.tensor(dw),atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(b.grad,upstream.sum((0,1)))

    def test_independent_protected_forward_physical_tiles(self):
        c=cfg(x_bits=3,w_bits=3,y_bits=3,x_noise=True,w_noise=True,y_noise=True)
        mask=torch.tensor([False,True,False,True,False,False,True])
        rows=self.x.reshape(-1,7).tolist(); weights=self.w.tolist()
        expected=[[sum(row[j]*wr[j] for j in range(7) if mask[j])+self.b[n].item()
                   for n,wr in enumerate(weights)] for row in rows]
        seen=[]
        for k in range(0,7,3):
            kl=min(3,7-k)
            xx=[scalar_q([0. if mask[k+j] else row[k+j] for j in range(kl)],3) for row in rows]
            for o in range(0,5,3):
                ol=min(3,5-o)
                ww=[scalar_q(w[k:k+kl],3) for w in weights[o:o+ol]]
                for m,row in enumerate(xx):
                    partial=[sum((row[j]+prescribed('x',m*kl+j))*(wr[j]+prescribed('w',n*kl+j))
                                 for j in range(kl) if not mask[k+j]) for n,wr in enumerate(ww)]
                    out=scalar_q([v+prescribed('y',m*ol+n) for n,v in enumerate(partial)],3)
                    for n,v in enumerate(out):expected[m][o+n]+=v
        with patch('photonic_vmm.core.draw_noise',side_effect=injected_noise):
            y=simulate_linear(self.x,self.w,self.b,c,protection=lambda _:mask,
                              trace=lambda k,o,p,a:seen.append((k,o)))
        torch.testing.assert_close(y,torch.tensor(expected).reshape_as(y),atol=2e-5,rtol=2e-5)
        self.assertEqual(seen,[(0,0),(0,3),(3,0),(3,3),(6,0),(6,3)])

    def test_none_all_protection_and_gradients(self):
        for mode in ('zero','legacy_abs'):
            c=dataclasses.replace(cfg(x_bits=3,w_bits=3,y_bits=3,x_noise=True,w_noise=True,y_noise=True),constant_range_mode=mode)
            y=simulate_linear(self.x,self.w,self.b,c)
            none=simulate_linear(self.x,self.w,self.b,c,protection=lambda x:torch.zeros_like(x,dtype=torch.bool))
            torch.testing.assert_close(none,y,rtol=0,atol=0)
            x=self.x.clone().requires_grad_();w=self.w.clone().requires_grad_()
            full=simulate_linear(x,w,self.b,c,protection=lambda x:torch.ones_like(x,dtype=torch.bool))
            digital=torch.nn.functional.linear(x,w,self.b)
            torch.testing.assert_close(full,digital,rtol=0,atol=0)
            grad=torch.randn_like(full)
            a=torch.autograd.grad(full,(x,w),grad);b=torch.autograd.grad(digital,(x,w),grad)
            for u,v in zip(a,b):torch.testing.assert_close(u,v)

    def test_policy_threshold_dynamic_and_empty(self):
        x=torch.tensor([[[6.,7.,1.],[1.,2.,8.]]])
        p=MaxAbsPolicy(6.,'batch_sequence')
        self.assertEqual(p(x).tolist(),[False,True,True])
        self.assertFalse(p(x*.1).any())
        self.assertEqual(MaxAbsPolicy(6.,'token')(x).tolist(),[[[False,True,False],[False,False,True]]])
        empty=torch.zeros(0,2,7,requires_grad=True)
        y=simulate_linear(empty,self.w,self.b,cfg(),protection=p)
        y.sum().backward();self.assertEqual(empty.grad.shape,empty.shape)
        with self.assertRaises(ValueError):simulate_linear(self.x,self.w,self.b,cfg(),protection=lambda x:torch.ones(7))

    def test_short_hat_optimization(self):
        # Synthetic regression is a training smoke test, not an LLM accuracy claim.
        x=torch.randn(24,7); target=torch.nn.functional.linear(x,torch.randn(5,7),torch.randn(5))
        c=cfg(x_bits=8,w_bits=8,y_bits=8,x_noise=True,w_noise=True,y_noise=True,x_std=.002,w_std=.002,y_std=.002)
        model=PhotonicLinear(torch.nn.Linear(7,5),c,'regression',MaxAbsPolicy(1.8,'batch_sequence'))
        opt=torch.optim.SGD(model.parameters(),lr=.15)
        losses=[]
        for _ in range(80):
            opt.zero_grad();loss=(model(x)-target).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            opt.step();losses.append(loss.item())
        self.assertLess(losses[-1],losses[0]*.05)
        print(f'HAT regression MSE: {losses[0]:.8f} -> {losses[-1]:.8f}')

    def test_dynamic_adc_width_one_and_two(self):
        from photonic_vmm.core import quantize
        for t in (torch.tensor([[.37],[8.2]]),torch.tensor([[.13,2.71],[-8.3,.14]])):
            torch.testing.assert_close(quantize(t,2),t,rtol=1e-6,atol=1e-6)

    def test_output_execution_modes_share_x_only_and_replay(self):
        from photonic_vmm.core import VMMConfig, draw_noise
        self.assertEqual(VMMConfig(constant_range_mode='zero').output_tile_execution, 'spatial_parallel')
        c=cfg(x_noise=True,w_noise=True,y_noise=True)
        d=dataclasses.replace(c,output_tile_execution='temporal_multiplexed')
        x=torch.tensor([[0.,1.,2.]]);w=torch.tensor([[0.,1.,2.],[2.,3.,4.],[4.,5.,6.]])
        self.assertTrue(torch.equal(draw_noise(x,c,'x','l',2,0,0),draw_noise(x,c,'x','l',2,0,3)))
        self.assertFalse(torch.equal(draw_noise(x,d,'x','l',2,0,0),draw_noise(x,d,'x','l',2,0,3)))
        for stage in ('w','y'):
            self.assertTrue(torch.equal(draw_noise(w,c,stage,'l',2,0,3),draw_noise(w,d,stage,'l',2,0,3)))
        for config in (c,d):
            a=simulate_linear(self.x,self.w,None,config,forward_index=2)
            b=simulate_linear(self.x,self.w,None,config,forward_index=2)
            self.assertTrue(torch.equal(a,b))
