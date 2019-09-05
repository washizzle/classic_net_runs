from PIL import Image
import numpy as np

class MNISTColor(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dataset_depth=1):
        super().__init__(self, root, train, transform, target_transform, download)
        self.dataset_depth = 1

    def __getitem__(self, item):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.dataset_depth == 1:
            img = Image.fromarray(img.numpy(), mode='RGB')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target