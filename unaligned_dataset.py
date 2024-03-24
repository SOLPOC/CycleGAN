import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class UnalignedDataset(Dataset):
    """
    This datasets class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the datasets flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def __init__(self, option, unaligned=False, mode='train'):
        """
        Initialize this datasets class.

        :param option: stores all the experiment flags
        """

        self.transform = transforms.Compose()
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(option.dataroot, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(option.dataroot, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        """
        Get picture from domain A by index, and get picture from domain B randomly
        :param index:
        :return: a dict con
        """

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

