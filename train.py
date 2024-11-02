import argparse
import os
from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple, Union

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.distributions.normal import Normal
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.classifier import TwoLayerNet
from utils.datasets import select_data_and_labels
from utils.interfaces import DataInterfaceReal, DataInterfaceSynthetic
from utils.utils import (all_elements_in_targets, freeze, save_dict_as_files,
                         set_seed, to_cpu)


class _TrainingInterface:
    @staticmethod
    def gen_rand_labels(n: int, device: torch.device) -> Tensor:
        return 2 * torch.randint(0, 2, (n,), device=device) - 1
    
    @abstractmethod
    def gen_data_and_labels(self, train: bool, device: torch.device) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    @staticmethod
    def identity(outs: Tensor, labels: Tensor) -> Tensor:
        assert len(outs.shape) == len(labels.shape) == 1
        assert all_elements_in_targets(labels, [-1, 1])
        return - outs * labels

    @classmethod
    def logistic(cls, outs: Tensor, labels: Tensor) -> Tensor:
        return torch.log(1 + cls.identity(outs, labels).exp())
    
    @staticmethod
    def train(
        lit: LightningLite,
        classifier: TwoLayerNet, 
        data: Tensor, 
        labels: Tensor, 
        loss_func: Callable[[Tensor, Tensor], Tensor],
        lr: float,
        epochs: int,
    ) -> float: # type: ignore

        classifier.train()
        optim = SGD(classifier.parameters(), lr, .9)
        scheduler = ReduceLROnPlateau(optim)

        classifier, optim = lit.setup(classifier, optim)
        
        for _ in tqdm(range(epochs), mininterval=100):
            optim.zero_grad(True)

            outs = classifier(data)
            loss = loss_func(outs, labels).mean()

            lit.backward(loss)
            optim.step()
            scheduler.step(loss)

        return loss.item()

    @staticmethod
    @torch.no_grad()
    def test(classifier: TwoLayerNet, data: Tensor, labels: Tensor) -> float:
        assert all_elements_in_targets(labels, [-1, 1])
        freeze(classifier)
        classifier.eval()
        return ((classifier(data) * labels) > 0).float().mean().item()

    @staticmethod
    def gen_perturbations(
        classifier: TwoLayerNet, 
        data: Tensor, 
        target_labels: Tensor, 
        loss_func: Callable[[Tensor, Tensor], Tensor],
        perturbation_size: float,
    ) -> Tensor:
        
        freeze(classifier)
        classifier.eval()

        data = data.detach().requires_grad_(True)
        outs = classifier(data)
        loss = loss_func(outs, target_labels).sum()
        grads, = torch.autograd.grad(loss, data)

        flatten_grads = grads.flatten(1)
        normalized_flatten_grads = torch.nn.functional.normalize(flatten_grads)
        normalized_grads = normalized_flatten_grads.view(grads.shape)
        perturbations = normalized_grads * perturbation_size

        return - perturbations

    @staticmethod
    def calc_agreement(data: Tensor, classifier: TwoLayerNet, adv_classifier: TwoLayerNet) -> float:
        signs = classifier(data).sign()
        adv_signs = adv_classifier(data).sign()
        agreement = signs == adv_signs
        return agreement.float().mean().item()
    
    @staticmethod
    def del_unnecessary_contents(s: Dict[str, Any], for_cossim: bool, for_map: bool) -> None:
        if (not for_cossim) and (not for_map):
            del s['classifier'], s['data'], s['labels'], s['advs'], s['adv_classifier']
        elif for_cossim:
            del s['classifier'], s['adv_classifier']
    

class TrainingInterfaceSynthetic(_TrainingInterface, DataInterfaceSynthetic):
    def gen_data_and_labels(self, train: bool, device: torch.device) -> Tuple[Tensor, Tensor]:
        n = self.n_sample if train else 1000
        assert n % 2 == 0, n

        if self.data_gen_method == 'gauss':
            data = torch.normal(0, 1, (n, self.in_dim), device=device)
            labels = self.gen_rand_labels(n, device)
        
        elif self.data_gen_method == 'shifted_gauss':
            half_n = n // 2
            mean = torch.full((self.in_dim,), .3, device=device)
            std = torch.ones(self.in_dim, device=device)
            pos = Normal(mean, std).sample((half_n,)) # type: ignore
            neg = Normal(-mean, std).sample((half_n,)) # type: ignore
            data = torch.vstack([pos, neg])
            labels = torch.cat((
                torch.ones(half_n, device=device), 
                -torch.ones(half_n, device=device)
            ))
        
        else:
            raise ValueError(self.data_gen_method)
        
        return data, labels
    

class TrainingInterfaceReal(_TrainingInterface, DataInterfaceReal):
    def gen_data_and_labels(self, train: bool, device: torch.device) -> Tuple[Tensor, Tensor]:
        return select_data_and_labels(self.dataset_root, self.dataset_name, train, device)
    
    def del_unnecessary_contents(self, s: Dict[str, Any], for_cossim: bool, for_map: bool) -> None:
        super().del_unnecessary_contents(s, for_cossim, for_map)
        if for_cossim or for_map:
            del s['data'], s['labels']


class Main(LightningLite):
    def run(
        self, 
        i: Union[TrainingInterfaceSynthetic, TrainingInterfaceReal],
        for_cossim: bool,
        for_map: bool,
    ) -> None:

        if os.path.exists(i.path):
            if (not for_cossim) and (not for_map):
                print(f'already exist (acc): {i.path}')
                return
            elif for_cossim and os.path.exists(os.path.join(i.path, 'advs')):
                print(f'already exist (cossim): {i.path}')
                return
            elif for_map \
            and os.path.exists(os.path.join(i.path, 'classifier')) \
            and os.path.exists(os.path.join(i.path, 'adv_classifier')):
                print(f'already exist (map): {i.path}')
                return
        else:
            os.makedirs(i.path, exist_ok=True)

        set_seed(i.seed)

        classifier = TwoLayerNet(i.in_dim, i.hidden_dim, i.slope)

        data, labels = i.gen_data_and_labels(True, self.device)

        loss_func = i.identity if i.loss_name == 'identity' else i.logistic

        loss = i.train(self, classifier, data, labels, loss_func, i.lr_1, i.epochs_1)
        acc = i.test(classifier, data, labels)

        target_labels = i.gen_rand_labels(len(data), self.device)
        perturbations = i.gen_perturbations(classifier, data, target_labels, loss_func, i.perturbation_size)

        advs = data + perturbations if i.on_original else perturbations

        adv_classifier = TwoLayerNet(i.in_dim, i.hidden_dim, i.slope)

        adv_loss = i.train(self, adv_classifier, advs, target_labels, loss_func, i.lr_2, i.epochs_2)
        #adv_acc = i.test(adv_classifier, advs, target_labels)
        adv_acc_for_natural = i.test(adv_classifier, data, labels)

        data_for_agreement = i.gen_data_and_labels(False, self.device)[0]
        agreement = i.calc_agreement(data_for_agreement, classifier, adv_classifier)

        save_data = {
            'classifier': classifier, # map
            'data': data, # cossim, map
            'labels': labels, # cossim, map
            #'loss': loss,
            'acc': acc, # acc
            #'target_labels': target_labels,
            'advs': advs, # cossim, map
            'adv_classifier': adv_classifier, # map
            #'adv_loss': adv_loss,
            #'adv_acc': adv_acc,
            'adv_acc_for_natural': adv_acc_for_natural, # acc
            #'data_for_agreement': data_for_agreement,
            'agreement': agreement, # acc
        }
        i.del_unnecessary_contents(save_data, for_cossim, for_map)
        to_cpu(save_data)
        save_dict_as_files(save_data, i.path)


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
    parser.add_argument('--on_original', '-o', action='store_true')
    parser.add_argument('--for_cossim', '-c', action='store_true')
    parser.add_argument('--for_map', '-m', action='store_true')

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
        i = TrainingInterfaceSynthetic(**interface_kwargs)

    elif args.mode == 'real':
        interface_kwargs.update({
            'dataset_name': args.dataset_name,
            'dataset_root': os.path.join('..', 'datasets')
        })
        i = TrainingInterfaceReal(**interface_kwargs)

    Main(**lite_kwargs).run(i, args.for_cossim, args.for_map)