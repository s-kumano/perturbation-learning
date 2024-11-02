import argparse
import os
from pathlib import Path
from typing import Dict, Sequence, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.lite import LightningLite
from torch import Tensor

from utils.classifier import TwoLayerNet
from utils.interfaces import (DataInterfaceReal, DataInterfaceSynthetic,
                              ResultReal, ResultSynthetic)
from utils.meshgrid import get_meshgrid_vector, get_square_meshgrid
from utils.theory import hat_f, hat_g
from utils.utils import at_least_one_element_in_targets, to_cpu


class _MapInterface:
    @staticmethod
    def calc(
        X: Tensor,
        y: Tensor,
        R: Tensor,
        gamma: float, 
        classifier_f: TwoLayerNet,
        classifier_g: TwoLayerNet,
        limits: Sequence[float],
        resolution: int,
        batch_size: int,
        samplings: int,
    ) -> Dict[str, Tensor]:
        
        resolution2 = resolution ** 2
        device = X.device

        axis_1 = F.normalize(X[y == 1].mean(0), dim=0) # (d,)
        axis_2 = F.normalize(X[y == -1].mean(0), dim=0) # (d,)
        
        # ((resolution, resolution), (resolution, resolution))
        meshgrid_x, meshgrid_y = get_square_meshgrid(resolution, limits, torch.device('cpu'))

        # (resolution, resolution, d)
        meshgrid_vector = get_meshgrid_vector(axis_1.cpu(), axis_2.cpu(), meshgrid_x, meshgrid_y)

        # (resolution*resolution, d)
        meshgrid_vector_flatten = meshgrid_vector.view(resolution2, -1)

        agreements = torch.full((resolution2,), float('inf'))
        hat_f_out = torch.full((resolution2,), float('inf'))
        hat_g_out = torch.full((resolution2,), float('inf'))

        ms = meshgrid_vector_flatten.split(batch_size)
        ags = agreements.split(batch_size)
        fs = hat_f_out.split(batch_size)
        gs = hat_g_out.split(batch_size)

        for m, ag, f, g in zip(ms, ags, fs, gs):
            m = m.to(device)
            ag[:] = (classifier_f(m).sign() == classifier_g(m).sign()).cpu()
            f[:] = hat_f(X, y, m, gamma, samplings).cpu()
            g[:] = hat_g(X, y, m, R, gamma, samplings).cpu()

        assert not at_least_one_element_in_targets(agreements, [float('inf')])
        assert not at_least_one_element_in_targets(hat_f_out, [float('inf')])
        assert not at_least_one_element_in_targets(hat_g_out, [float('inf')])

        agreements = agreements.view(resolution, resolution)
        hat_f_out = hat_f_out.view(resolution, resolution)
        hat_g_out = hat_g_out.view(resolution, resolution)

        projected_X = X @ torch.stack([axis_1, axis_2]).T

        return {
            'meshgrid_x': meshgrid_x, # (resolution, resolution)
            'meshgrid_y': meshgrid_y, # (resolution, resolution)
            'agreements': agreements, # (resolution, resolution)
            'hat_f': hat_f_out, # (resolution, resolution)
            'hat_g': hat_g_out, # (resolution, resolution)
            'projected_X': projected_X, # (N, 2)
        }


class MapInterfaceSynthetic(_MapInterface, DataInterfaceSynthetic):
    def result(self, device: torch.device) -> ResultSynthetic:
        return ResultSynthetic(
            'data',
            self.in_dim,
            self.hidden_dim,
            self.slope,
            self.loss_name,
            self.epochs_1,
            self.epochs_2,
            self.perturbation_size,
            self.seed,
            self.on_original,
            self.lr_1,
            self.lr_2,
            device,
            self.data_gen_method,
            self.n_sample,
        )
    

class MapInterfaceReal(_MapInterface, DataInterfaceReal):
    def result(self, device: torch.device) -> ResultReal:
        return ResultReal(
            'data',
            self.hidden_dim,
            self.slope,
            self.loss_name,
            self.epochs_1,
            self.epochs_2,
            self.perturbation_size,
            self.seed,
            self.on_original,
            self.lr_1,
            self.lr_2,
            device,
            self.dataset_name,
            self.dataset_root,
        )


class Main(LightningLite):
    def run(
        self, 
        i: Union[MapInterfaceSynthetic, MapInterfaceReal],
        limits: Sequence[float],
        resolution: int,
        batch_size: int,
        samplings: int,
    ) -> None:

        path = os.path.join(i.path, 'map')

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            Path(path).touch()
        
        r = i.result(self.device)

        map_info = i.calc(
            r.data,
            r.labels,
            r.advs,
            i.slope,
            r.classifier,
            r.adv_classifier,
            limits,
            resolution,
            batch_size,
            samplings,
        )
        to_cpu(map_info)
        torch.save(map_info, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hidden_dim', type=int)
    parser.add_argument('slope', type=float)
    parser.add_argument('loss_name', choices=('identity', 'logistic'))
    parser.add_argument('lr_1', type=float)
    parser.add_argument('lr_2', type=float)
    parser.add_argument('epochs_1', type=int)
    parser.add_argument('epochs_2', type=int)
    parser.add_argument('perturbation_size', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('limits', nargs=2, type=float)
    parser.add_argument('device', type=int)
    parser.add_argument('--on_original', '-o', action='store_true')
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=160000)
    parser.add_argument('--samplings', type=int, default=100)

    subparsers = parser.add_subparsers(dest='mode', required=True)

    parser_synthetic = subparsers.add_parser('synthetic')
    parser_synthetic.add_argument('in_dim', type=int)
    parser_synthetic.add_argument('data_gen_method', choices=('gauss', 'shifted_gauss'))
    parser_synthetic.add_argument('n_sample', type=int)

    parser_real = subparsers.add_parser('real')
    parser_real.add_argument('dataset_name', choices=('MNIST', 'FMNIST'))

    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'strategy': 'ddp_find_unused_parameters_false',
        'devices': [args.device],
        'precision': 16,
    }

    interface_kwargs = {
        'data_root': 'data',
        'hidden_dim': args.hidden_dim,
        'slope': args.slope,
        'loss_name': args.loss_name,
        'epochs_1': args.epochs_1,
        'epochs_2': args.epochs_2,
        'perturbation_size': args.perturbation_size,
        'seed': args.seed,
        'on_original': args.on_original,
        'lr_1': args.lr_1,
        'lr_2': args.lr_2,
    }

    if args.mode == 'synthetic':
        interface_kwargs.update({
            'in_dim': args.in_dim,
            'data_gen_method': args.data_gen_method,
            'n_sample': args.n_sample,
        })
        i = MapInterfaceSynthetic(**interface_kwargs)

    elif args.mode == 'real':
        interface_kwargs.update({
            'dataset_name': args.dataset_name,
            'dataset_root': os.path.join('..', 'datasets')
        })
        i = MapInterfaceReal(**interface_kwargs)

    Main(**lite_kwargs).run(i, args.limits, args.resolution, args.batch_size, args.samplings)