from typing import Any, Callable, Literal, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset

from .utils import dataloader


def select_data_and_labels(
    root: str, 
    dname: Literal['MNIST', 'FMNIST'], 
    train: bool,
    device: torch.device = torch.device('cpu'),
) -> Tuple[Tensor, Tensor]:
    
    dataset_cls = MNIST if dname == 'MNIST' else FMNIST
    dataset = dataset_cls(root, train)
    
    loader = dataloader(dataset, len(dataset), False, 3, False)
    data, labels = next(iter(loader))

    pos_class, neg_class = (1, 2) if dname == 'MNIST' else (0, 9)
    indices = (labels == pos_class) | (labels == neg_class)
    data, labels = data[indices], labels[indices]
    labels = torch.where(labels == pos_class, 1, -1)

    data -= dataset_cls.mean[0]

    return data.to(device), labels.to(device)


class SequenceDataset(Dataset):
    def __init__(
        self, 
        xs: Any, 
        ys: Any, 
        x_transform: Optional[Callable] = None, 
        y_transform: Optional[Callable] = None,
    ) -> None:
        assert len(xs) == len(ys)
        self.xs, self.ys = xs, ys
        self.x_transform, self.y_transform = x_transform, y_transform

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.x_transform(self.xs[idx]) if self.x_transform else self.xs[idx]
        y = self.y_transform(self.ys[idx]) if self.y_transform else self.ys[idx]
        return x, y
    

# `mean` and `std` must be list (not tuple)
# to be compatible with the type hint of `torchvision.transforms.functional.normalize`.
    

class MNIST(torchvision.datasets.MNIST):
    mean = [.1307]
    std = [.3081]
    n_class = 10
    classes = tuple(range(10))
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = transform if transform else T.ToTensor()
        super().__init__(root, train, transform, target_transform, True)


class FMNIST(torchvision.datasets.FashionMNIST):
    mean = [.2860]
    std = [.3530]
    n_class = 10
    #classes = ('T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    #           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    classes = ('Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = transform if transform else T.ToTensor()
        super().__init__(root, train, transform, target_transform, True)


class CIFAR10(torchvision.datasets.CIFAR10):
    # This must be list (not tuple)
    # to be compatible with the type hint of `torchvision.transforms.functional.normalize`.
    mean = [.4914, .4822, .4465]
    std = [.2470, .2435, .2616]
    n_class = 10
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    size = (3, 32, 32)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        if transform is None:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]) if train else T.ToTensor()

        super().__init__(root, train, transform, target_transform, True)