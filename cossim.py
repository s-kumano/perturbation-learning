import argparse
import os
from abc import abstractmethod
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from pytorch_lightning.lite import LightningLite
from torch import Tensor

from utils.interfaces import (DataInterfaceReal, DataInterfaceSynthetic,
                              ResultReal, ResultSynthetic)
from utils.theory import direction


class _CossimInterface:
    @staticmethod
    def calc(
        data: Tensor, 
        labels: Tensor, 
        perturbations: Tensor, 
        gamma: float,
        batch_size: int, 
        samplings: int, 
    ) -> float:
        '''
        Args:
            data: (N, d) := (the number of data, dim)
            labels: (N,) := (the number of data)
            perturbations: (N, d) := (the number of data, dim)
        
        Returns:
            direction: (N, d)
        '''

        bs = batch_size

        if bs == -1:
            bs = len(data)

        results = torch.empty(*data.shape, device=data.device)
        for d, l, r in zip(data.split(bs), labels.split(bs), results.split(bs)):
            r[:] = direction(d, l, gamma, samplings)

        normalized_perturbations = F.normalize(perturbations)

        return (results * normalized_perturbations).sum(1).abs().mean().item()
    
    @abstractmethod
    def result(self) -> Union[ResultSynthetic, ResultReal]:
        pass


class CossimInterfaceSynthetic(_CossimInterface, DataInterfaceSynthetic):
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
            False,
            self.lr_1,
            self.lr_2,
            device,
            self.data_gen_method,
            self.n_sample,
        )
    

class CossimInterfaceReal(_CossimInterface, DataInterfaceReal):
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
            False,
            self.lr_1,
            self.lr_2,
            device,
            self.dataset_name,
            self.dataset_root,
        )


class Main(LightningLite):
    def run(
        self, 
        i: Union[CossimInterfaceSynthetic, CossimInterfaceReal], 
        batch_size: int, 
        samplings: int,
    ) -> None:

        path = os.path.join(i.path, 'cossim')

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            Path(path).touch()
        
        r = i.result(self.device)

        cossim = i.calc(
            r.data,
            r.labels,
            r.advs,
            i.slope,
            batch_size,
            samplings,
        )
        
        torch.save(cossim, path)


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
    parser.add_argument('device', type=int)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--samplings', type=int, default=1000)

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
        'on_original': False,
        'lr_1': args.lr_1,
        'lr_2': args.lr_2,
    }

    if args.mode == 'synthetic':
        interface_kwargs.update({
            'in_dim': args.in_dim,
            'data_gen_method': args.data_gen_method,
            'n_sample': args.n_sample,
        })
        i = CossimInterfaceSynthetic(**interface_kwargs)

    elif args.mode == 'real':
        interface_kwargs.update({
            'dataset_name': args.dataset_name,
            'dataset_root': os.path.join('..', 'datasets')
        })
        i = CossimInterfaceReal(**interface_kwargs)

    Main(**lite_kwargs).run(i, args.batch_size, args.samplings)