import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, OrderedDict, Tuple

import torch
from torch import Tensor

from .classifier import TwoLayerNet
from .datasets import select_data_and_labels
from .utils import freeze


@dataclass
class _BaseInterface:
    data_root: str
    dname: str
    in_dim: int
    hidden_dim: int
    slope: float
    loss_name: Literal['identity', 'logistic']
    epochs_1: int
    epochs_2: int
    perturbation_size: float
    seed: int
    on_original: bool


@dataclass
class _SyntheticInterface:
    data_gen_method: Literal['gauss', 'shifted_gauss']
    n_sample: int

    dname: str = field(default='synthetic', init=False)


@dataclass
class _RealInterface:
    dataset_name: Literal['MNIST', 'FMNIST']
    dataset_root: str

    dname: str = field(default='real', init=False)
    in_dim: int = field(default=784, init=False)


@dataclass
class _DataInterface(_BaseInterface):
    lr_1: float
    lr_2: float

    @property
    def id(self) -> str:
        raise NotImplementedError
    
    @property
    def path(self) -> str:
        return os.path.join(self.data_root, self.dname, self.id)


@dataclass
class DataInterfaceSynthetic(_SyntheticInterface, _DataInterface):
    @property
    def id(self) -> str:
        return f'{self.in_dim}_{self.hidden_dim}_{self.slope}_{self.data_gen_method}' \
               f'_{self.n_sample}_{self.loss_name}_{self.lr_1}_{self.lr_2}_{self.epochs_1}_{self.epochs_2}' \
               f'_{self.perturbation_size}_{self.seed}_{self.on_original}'


@dataclass
class DataInterfaceReal(_RealInterface, _DataInterface):
    @property
    def id(self) -> str:
        return f'{self.dataset_name}_{self.hidden_dim}_{self.slope}_{self.loss_name}_{self.lr_1}' \
               f'_{self.lr_2}_{self.epochs_1}_{self.epochs_2}_{self.perturbation_size}' \
               f'_{self.seed}_{self.on_original}'


@dataclass
class _Result(_DataInterface):
    device: torch.device

    def _load(self, key: str) -> Any:
        _key = '_' + key
        if not hasattr(self, _key):
            p = os.path.join(self.path, key)
            obj = torch.load(p, map_location=self.device)
            setattr(self, _key, obj)
        return getattr(self, _key)
    
    @property
    def data(self) -> Tensor:
        raise NotImplementedError
    
    @property
    def labels(self) -> Tensor:
        raise NotImplementedError
    
    @property
    def advs(self) -> Tensor:
        return self._load('advs')
    
    @property
    def target_labels(self) -> Tensor:
        return self._load('target_labels')
    
    @property
    def acc(self) -> float:
        return self._load('acc')
    
    @property
    def adv_acc(self) -> float:
        return self._load('adv_acc')

    @property
    def adv_acc_for_natural(self) -> float:
        return self._load('adv_acc_for_natural')
    
    @property
    def agreement(self) -> float:
        return self._load('agreement')
    
    @property
    def cossim(self) -> float:
        return self._load('cossim')
    
    @property
    def map(self) -> Dict[str, Tensor]:
        return self._load('map')
    
    @property
    def meshgrid_x(self) -> Tensor:
        return self.map['meshgrid_x']
    
    @property
    def meshgrid_y(self) -> Tensor:
        return self.map['meshgrid_y']
    
    @property
    def agreements(self) -> Tensor:
        return self.map['agreements']
    
    @property
    def hat_f(self) -> Tensor:
        return self.map['hat_f']
    
    @property
    def hat_g(self) -> Tensor:
        return self.map['hat_g']
    
    @property
    def projected_X(self) -> Tensor:
        return self.map['projected_X']
    
    @property
    def classifier(self) -> TwoLayerNet:
        return self._access_classifier('classifier')
    
    @property
    def adv_classifier(self) -> TwoLayerNet:
        return self._access_classifier('adv_classifier')
    
    def _access_classifier(self, key: Literal['classifier', 'adv_classifier']) -> TwoLayerNet:
        _key = '_' + key
        if not hasattr(self, _key):
            p = os.path.join(self.path, key)
            params = torch.load(p, map_location=self.device)
            setattr(self, _key, self._setup_classifier(params))
        return getattr(self, _key)
    
    def _setup_classifier(self, params: OrderedDict) -> TwoLayerNet:
        classifier = TwoLayerNet(self.in_dim, self.hidden_dim, self.slope)
        classifier.to(self.device)
        classifier.load_state_dict(params)
        classifier.eval()
        freeze(classifier)
        return classifier
    

@dataclass
class _ResultsAlongLrs(_BaseInterface):
    lrs_1: Tuple[float, ...]
    lrs_2: Tuple[float, ...]
    device: torch.device
    
    def result(self, lr_1: float, lr_2) -> _Result:
        raise NotImplementedError
    
    @property
    def results(self) -> List[_Result]:
        if not hasattr(self, '_results'):
            self._results = [self.result(lr_1, lr_2) for lr_1 in self.lrs_1 for lr_2 in self.lrs_2]
        return self._results
    
    def _gather(self, _key: str, key: str) -> List[float]:
        if not hasattr(self, _key):
            setattr(self, _key, [getattr(r, key) for r in self.results])
        return getattr(self, _key)
    
    @property
    def accs(self) -> List[float]:
        return self._gather('_accs', 'acc')
    
    @property
    def adv_accs(self) -> List[float]:
        return self._gather('_adv_accs', 'adv_acc')
    
    @property
    def adv_accs_for_natural(self) -> List[float]:
        return self._gather('_adv_accs_for_natural', 'adv_acc_for_natural')
    
    @property
    def agreements(self) -> List[float]:
        return self._gather('_agreements', 'agreement')
    
    @property
    def cossims(self) -> List[float]:
        return self._gather('_cossims', 'cossim')
    
    @property
    def acc(self) -> float:
        return max(self.accs)
    
    @property
    def adv_acc(self) -> float:
        return max(self.adv_accs)
    
    @property
    def adv_acc_for_natural(self) -> float:
        return max(self.adv_accs_for_natural)
    
    @property
    def agreement(self) -> float:
        return max(self.agreements)
    
    @property
    def cossim(self) -> float:
        return max(self.cossims)
    

@dataclass
class ResultSynthetic(DataInterfaceSynthetic, _Result):
    @property
    def data(self) -> Tensor:
        return self._load('data')
    
    @property
    def labels(self) -> Tensor:
        return self._load('labels')
    

@dataclass
class ResultsSyntheticAlongLrs(_SyntheticInterface, _ResultsAlongLrs):
    def result(self, lr_1: float, lr_2: float) -> ResultSynthetic:
        return ResultSynthetic(
            self.data_root, 
            self.in_dim, 
            self.hidden_dim,
            self.slope,
            self.loss_name,
            self.epochs_1,
            self.epochs_2,
            self.perturbation_size,
            self.seed,
            self.on_original,
            lr_1,
            lr_2,
            self.device,
            self.data_gen_method,
            self.n_sample,
        )


@dataclass
class ResultReal(DataInterfaceReal, _Result):
    @property
    def data_and_labels(self) -> Tuple[Tensor, Tensor]:
        if not hasattr(self, '_data_and_labels'):
            self._data, self._labels \
                = select_data_and_labels(self.dataset_root, self.dataset_name, True, self.device)
            self._data = self._data.flatten(1)
        return self._data, self._labels
    
    @property
    def data(self) -> Tensor:
        return self.data_and_labels[0]
    
    @property
    def labels(self) -> Tensor:
        return self.data_and_labels[1]
    
    @property
    def advs(self) -> Tensor:
        return super().advs.flatten(1)


@dataclass
class ResultsRealAlongLrs(_RealInterface, _ResultsAlongLrs):
    def result(self, lr_1: float, lr_2: float) -> ResultReal:
        return ResultReal(
            self.data_root, 
            self.hidden_dim,
            self.slope,
            self.loss_name,
            self.epochs_1,
            self.epochs_2,
            self.perturbation_size,
            self.seed,
            self.on_original,
            lr_1,
            lr_2,
            self.device,
            self.dataset_name,
            self.dataset_root,
        )