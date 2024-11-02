import os
import random
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchvision.transforms.functional import normalize
from tqdm import tqdm


def set_seed(seed: int = 0) -> None:
    #from lightning_lite.utilities.seed import seed_everything
    #seed_everything(seed, True)
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PL_SEED_WORKERS'] = str(1)


def gpu(id: int) -> torch.device:
    print(torch.cuda.get_device_name(id))
    return torch.device(f'cuda:{id}')


def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d


def save_dict_as_files(dict: Dict[str, Any], root: str) -> None:
    for k, v in dict.items():
        p = os.path.join(root, k)
        torch.save(v, p)


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


@torch.no_grad()
def at_least_one_element_in_targets(x: Tensor, targets: List[float]) -> bool:
    return torch.isin(x, torch.tensor(targets, device=x.device)).any().item() # type: ignore


@torch.no_grad()
def all_elements_in_targets(x: Tensor, targets: List[float]) -> bool:
    return torch.isin(x, torch.tensor(targets, device=x.device)).all().item() # type: ignore


def freeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def get_model_device(model: Module) -> torch.device:
    return next(model.parameters()).device
        

def dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool, 
    num_workers: int = 3, 
    pin_memory: bool = True, 
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class ModelWithNormalization(Module):
    def __init__(self, model: Module, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.model = model
        self.mean, self.std = mean, std

    def forward(self, x: Tensor) -> Tensor:
        assert in_range(x, 0, 1)
        return self.model(normalize(x, self.mean, self.std))


class CalcClassificationAcc(LightningLite):
    def run(
        self, 
        classifier: Module, 
        loader: DataLoader, 
        n_class: int, 
        top_k: int = 1,
        average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro',
    ) -> Union[float, List[float]]:

        classifier = self.setup(classifier)
        loader = self.setup_dataloaders(loader) # type: ignore

        freeze(classifier)
        classifier.eval()

        metric = MulticlassAccuracy(n_class, top_k, average)
        self.to_device(metric)

        for xs, labels in tqdm(loader):
            outs = classifier(xs)
            metric(outs, labels)
        
        acc = metric.compute()
        return acc.tolist() if average == 'none' else acc.item()