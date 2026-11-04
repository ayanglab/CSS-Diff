"""CPU regression tests; no real data, weights, GPU or W&B service needed."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from models.auxiliary_weights import load_auxiliary_weights, validate_auxiliary_weights
from models.pss_generator import PSSGenerator
from options.train_options import TrainOptions

ROOT = Path(__file__).resolve().parents[1]


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_yaml_cli_precedence(self):
        with patch.object(sys, 'argv', ['train.py', '--config', str(ROOT/'train_configs/train_pss_templete.yaml'), '--name', 'cli', '--gpu_ids', '-1']):
            opt = TrainOptions().gather_options()
        self.assertEqual(opt.name, 'cli')
        self.assertEqual(opt.model, 'pss_lsc_sgp')
        self.assertTrue(opt.wdb_disabled)
        with self.assertRaises(FileNotFoundError):
            TrainOptions().load_yaml_options('/nonexistent/css.yaml')

    def test_preflight_reports_both_files(self):
        with self.assertRaisesRegex(FileNotFoundError, 'SGP.*LSC'):
            validate_auxiliary_weights(SimpleNamespace(sgp_weights='', lsc_weights=''), ['SGP', 'LSC'])

    def test_strict_loading_and_input_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pth'
            module = torch.nn.Conv2d(1, 1, 1)
            torch.save(module.state_dict(), path)
            load_auxiliary_weights(module, path, 'cpu', 'test')
            x = torch.ones(1,1,4,4,requires_grad=True)
            module(x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
            torch.save({'bad': torch.zeros(1)}, path)
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                load_auxiliary_weights(module, path, 'cpu', 'test')

    def test_trajectory_and_legacy_checkpoint(self):
        cfg = yaml.safe_load((ROOT/'train_configs/train_pss_templete.yaml').read_text())['PSS']
        cfg['image_size'] = 32
        cfg['model'].update(ch=32,ch_mult=[1,2],num_res_blocks=1,attn_resolutions=[],dropout=0.0)
        generator = PSSGenerator(cfg,device='cpu')
        self.assertNotIn('betas', generator.state_dict())
        generator.load_state_dict(generator.state_dict(), strict=True)
        x = torch.randn(1,1,32,32,requires_grad=True)
        out, trace = generator(x, seq=[0,2], cond=x, return_trajectory=True)
        self.assertEqual(trace['state_timesteps'], [2,0,-1])
        self.assertEqual(len(trace['states']), 3)
        self.assertTrue(torch.allclose(out,generator(x,seq=[0,2],cond=x)))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for seq in [[], [2,1], [-1], [50]]:
            with self.assertRaises(ValueError):
                generator(x,seq=seq,cond=x)

    def test_import_does_not_launch_sweep(self):
        with patch('wandb.sweep', side_effect=AssertionError('network call')), patch('wandb.agent', side_effect=AssertionError('network call')):
            import train
            import pss_train
            import test


if __name__ == '__main__':
    unittest.main()
