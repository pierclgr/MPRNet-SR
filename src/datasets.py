import os
from skimage import io
from torch.utils import data
import random
import torch
from src.augmentations import random_crop, random_90_rotation, random_horizontal_flip
import torchvision.transforms as T


class TrainDataset(data.Dataset):
    """
    PyTorch dataset loading a train dataset of lr and hr patches
    """

    def __init__(self, dataset_path: str, scales: list = None,
                 degradation: str = "bicubic", patch_size: int = None, augment: bool = True) -> None:
        """
        Constructor method of the class

        :param dataset_path: path of the folder containing dataset images (str)
        :param scales: list containing the resolution scales to consider (list, default None)
        :param degradation: type of degraded images to use (str, default "bicubic")
        :param patch_size: size of the square (patch_size x patch_size) lr patches to extract (int, default 64)
        """

        super(TrainDataset, self).__init__()

        # define dataset path
        self.dataset_path = dataset_path

        # define scales to use if not given
        if not scales:
            self.scales = [2, 3, 4]
        else:
            self.scales = scales

        # define default patch size to use
        if not patch_size:
            self.patch_size = 64
        else:
            self.patch_size = patch_size

        # define degradation method to use
        self.degradation = degradation.lower()

        # define augment
        self.augment = augment

        # extract the image file names from the dataset
        self.filenames = sorted(os.listdir(os.path.join(dataset_path, "hr")))

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: length of the dataset
        """

        return len(self.filenames)

    def __getitem__(self, item) -> tuple:
        """
        Get a HR image and the corresponding LR images in all the scales

        :param item: the chosen item index in the dataset
        """

        # select the image to pick
        file_name = self.filenames[item]

        # extract the HR image from the HR folder
        hr_image_path = os.path.join(self.dataset_path, "hr", file_name)
        hr_image = io.imread(hr_image_path)

        # define the output tuple as empty
        output_tuple = ()

        # extract the LR images from the LR folder
        for scale in self.scales:
            # extract the LR image for the current scale from the LR folder
            lr_image_path = os.path.join(self.dataset_path, "lr", self.degradation, "x" + str(scale),
                                         file_name)
            lr_image = io.imread(lr_image_path)

            # extract lr and hr patches
            lr_patch, hr_patch = random_crop(lr_image, hr_image, scale, self.patch_size)

            # if augmentation is required
            if self.augment:
                # flip the patches
                lr_patch, hr_patch = random_horizontal_flip(lr_patch, hr_patch)

                # rotate the patches
                lr_patch, hr_patch = random_90_rotation(lr_patch, hr_patch)

            # add the current scale_factor-LR-HR triple to the output tuple
            output_tuple += (scale, T.ToTensor()(lr_patch.copy()), T.ToTensor()(hr_patch.copy()))

        return output_tuple

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        Collate function for the selection of a batch of random scale

        :param batch: list containing the batch samples extracted from the dataset using its __getitem__ method (list)
        :return: tuple containing the LR and HR batches in the chosen scale
        """

        # unzip the batch
        unzipped = list(zip(*batch))

        # choose a random scale from the ones given for the current batch
        starting_sub_index = random.randint(0, int(len(unzipped) / 3) - 1) * 3
        scale = unzipped[starting_sub_index][0]

        # stack the hr and lr batches into a unique PyTorch tensor
        lr = torch.stack(unzipped[starting_sub_index + 1])
        hr = torch.stack(unzipped[starting_sub_index + 2])

        # return the batch
        return scale, lr, hr


class TestDataset(data.Dataset):
    """
    PyTorch dataset loading a test dataset of hr and lr images
    """

    def __init__(self, dataset_path: str, scales: list = None, degradation: str = "bicubic") -> None:
        """
        Constructor method of the class

        :param dataset_path: path of the folder containing dataset images (str)
        :param scales: list containing the resolution scales to consider (list, default None)
        :param degradation: type of degraded images to use (str, default "bicubic")
        """

        super(TestDataset, self).__init__()

        # define dataset path
        self.dataset_path = dataset_path

        # define scales to use if not given
        if not scales:
            self.scales = [2, 3, 4]
        else:
            self.scales = scales

        # define degradation method to use
        self.degradation = degradation.lower()

        # extract the image file names from the dataset
        self.filenames = sorted(os.listdir(os.path.join(dataset_path, "hr")))

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: length of the dataset
        """

        return len(self.filenames)

    def __getitem__(self, item) -> tuple:
        """
        Get a HR image and the corresponding LR images in all the scales

        :param item: the chosen item index in the dataset
        """

        # select the image to pick
        file_name = self.filenames[item]

        # extract the HR image from the HR folder
        hr_image_path = os.path.join(self.dataset_path, "hr", file_name)
        hr_image = io.imread(hr_image_path)

        # define the output tuple as empty
        output_tuple = ()

        # extract the LR images from the LR folder
        for scale in self.scales:
            # extract the LR image for the current scale from the LR folder
            lr_image_path = os.path.join(self.dataset_path, "lr", self.degradation, "x" + str(scale),
                                         file_name)
            lr_image = io.imread(lr_image_path)

            # add the current scale_factor-LR-HR triple to the output tuple
            output_tuple += (scale, T.ToTensor()(lr_image), T.ToTensor()(hr_image))

        return output_tuple
