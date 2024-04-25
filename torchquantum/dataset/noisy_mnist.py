"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import random
import numpy as np

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt


__all__ = ["NoisyMNIST", "NoisyMNISTDataset"]


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class NoisyMNISTDataset:
    def __init__(
        self,
        root: str,
        split: str,
        noise: str,
        train_valid_split_ratio: List[float],
        center_crop,
        resize,
        resize_mode,
        binarize,
        binarize_threshold,
        digits_of_interest,
        n_test_samples,
        n_valid_samples,
        fashion,
        n_train_samples,
        same_n_samples_each_class=False,
        std_dev = 0,
        seed = 42,
    ):
        self.root = root
        self.split = split
        self.noise = noise
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.n_train_samples = n_train_samples
        self.fashion = fashion

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.n_digits = len(digits_of_interest)

        self.same_n_samples_each_class = same_n_samples_each_class
        self.std_dev = std_dev

        self.load()
        self.n_instance = len(self.data)
        

    def _get_indices(self, subset, n_samples):
        sample_ctr = {}
        sample_quota = {}
        indices = []
        n_samples_each_class = n_samples // self.n_digits

        for digit_of_interest in self.digits_of_interest:
            sample_ctr[digit_of_interest] = 0
            if digit_of_interest == self.digits_of_interest[-1]:
                sample_quota[digit_of_interest] = (
                    n_samples - (self.n_digits - 1) * n_samples_each_class
                )
            else:
                sample_quota[digit_of_interest] = n_samples_each_class

        for idx in subset.indices:
            digit = subset.dataset[idx][1]
            if sample_ctr[digit] < sample_quota[digit]:
                indices.append(idx)
                sample_ctr[digit] += 1

        return indices

    def load(self):
        tran = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 28:
            tran.append(transforms.Resize(self.resize, interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            if self.fashion:
                train_valid = datasets.FashionMNIST(
                    self.root, train=True, download=True, transform=transform
                )
            else:
                train_valid = datasets.MNIST(
                    self.root, train=True, download=True, transform=transform
                )
            idx, _ = torch.stack(
                [train_valid.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                if self.n_train_samples is None:
                    # use all samples in train set
                    self.data = train_subset
                else:
                    if self.same_n_samples_each_class:
                        train_subset.indices = self._get_indices(
                            train_subset, self.n_train_samples
                        )
                    else:
                        train_subset.indices = train_subset.indices[
                            : self.n_train_samples
                        ]
                    self.data = train_subset
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_train_samples} images as "
                        f"TRAIN set."
                    )
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    if self.same_n_samples_each_class:
                        valid_subset.indices = self._get_indices(
                            valid_subset, self.n_valid_samples
                        )
                    else:
                        valid_subset.indices = valid_subset.indices[
                            : self.n_valid_samples
                        ]
                    self.data = valid_subset
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_valid_samples} images as "
                        f"VALID set."
                    )

        else:
            if self.fashion:
                test = datasets.FashionMNIST(
                    self.root, train=False, transform=transform
                )
            else:
                test = datasets.MNIST(self.root, train=False, transform=transform)
            idx, _ = torch.stack(
                [test.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                if self.same_n_samples_each_class:
                    sample_ctr = {}
                    sample_quota = {}
                    indices = []
                    n_samples_each_class = self.n_test_samples // self.n_digits

                    for digit_of_interest in self.digits_of_interest:
                        sample_ctr[digit_of_interest] = 0
                        if digit_of_interest == self.digits_of_interest[-1]:
                            sample_quota[digit_of_interest] = (
                                self.n_test_samples
                                - (self.n_digits - 1) * n_samples_each_class
                            )
                        else:
                            sample_quota[digit_of_interest] = n_samples_each_class

                    for idx, target in enumerate(test.targets):
                        digit = target.item()
                        if sample_ctr[digit] < sample_quota[digit]:
                            indices.append(idx)
                            sample_ctr[digit] += 1

                    test.targets = test.targets[indices]
                    test.data = test.data[indices]
                else:
                    test.targets = test.targets[: self.n_test_samples]
                    test.data = test.data[: self.n_test_samples]

                self.data = test
                logger.warning(
                    f"Only use the front {self.n_test_samples} " f"images as TEST set."
                )

    def snp(self, image, prob):
        # Extract the dimensions of the image tensor
        batch_size, height, width = image.shape
        
        # Create a mask for salt noise (1)
        salt_mask = torch.rand(batch_size, height, width) < prob/2
        
        # Create a mask for pepper noise (0)
        pepper_mask = torch.rand(batch_size, height, width) < prob/2
        
        # Create a mask for pixels to be left unchanged
        unchanged_mask = ~(salt_mask | pepper_mask)
        
        # Create a new tensor with salt and pepper noise applied
        noisy_image = image.clone()
        noisy_image[salt_mask] = 1
        noisy_image[pepper_mask] = 0
        noisy_image[unchanged_mask] = image[unchanged_mask]
        return noisy_image
    
    def poisson(self, image, strength):
        scale = (100-strength)/strength
        img = img.float() * scale
        lowest = min(torch.min(img), 0)
        img = torch.poisson(img-lowest) + lowest
        img = img/scale
        return img
    
    # Can optionally add noise before returning image at index
    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (
                img <= self.binarize_threshold
            )

        digit = self.digits_of_interest.index(self.data[index][1])

        if self.std_dev > 0:
            if self.noise == "gaussian":
                noise = torch.randn_like(img)
                img = img + self.std_dev * noise
            elif self.noise == "saltandpepper":
                if self.std_dev > 1:
                    raise InvalidArgumentException("S&P strength must be a percentage (between 0 and 1)")
                img = self.snp(img, self.std_dev)
            elif self.noise == "poisson":
                if self.std_dev > 1:
                    raise InvalidArgumentException("Poisson strength must be a percentage (between 0 and 1)")
                strength = 100*self.std_dev # I did the math in self.poisson using percentages
                img = self.poisson(img, strength)
            elif self.noise == "speckle":
                noise = torch.randn_like(img)
                img = img + img * self.std_dev * noise
                
            
        instance = {"image": img, "digit": digit}
        return instance

    def __len__(self) -> int:
        return self.n_instance

# Types of noise:
    # Gaussian: std_dev = standard deviation of Gaussian noise. Higher deviation = more noisy
    # Salt and pepper: std_dev = percentage of pixels turned to salt/pepper (0.5 chance of either)
    # Poisson: std_dev = factor multiplied to noise
    # Speckle: std_dev = standard deviation of Gaussian noise. Higher deviation = more noisy
class NoisyMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train_valid_split_ratio: List[float],
        noise="gaussian",
        center_crop=28,
        resize=28,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=tuple(range(10)),
        n_test_samples=None,
        n_valid_samples=None,
        fashion=False,
        n_train_samples=None,
        std_dev=0,
    ):
        self.root = root

        super().__init__(
            {
                split: NoisyMNISTDataset(
                    root=root,
                    split=split,
                    noise="gaussian",
                    train_valid_split_ratio=train_valid_split_ratio,
                    center_crop=center_crop,
                    resize=resize,
                    resize_mode=resize_mode,
                    binarize=binarize,
                    binarize_threshold=binarize_threshold,
                    digits_of_interest=digits_of_interest,
                    n_test_samples=n_test_samples,
                    n_valid_samples=n_valid_samples,
                    fashion=fashion,
                    n_train_samples=n_train_samples,
                    std_dev=std_dev,
                )
                for split in ["train", "valid", "test"]
            }
        )



