import os
from skimage import io
from torch.utils import data
import random
import torch
from src.augmentations import random_crop, random_90_rotation, random_horizontal_flip
import torchvision.transforms as T
import numpy as np
import cv2

cv2.setNumThreads(0)


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


class ValidationDataset(data.Dataset):
    """
    PyTorch dataset loading a train dataset of lr and hr patches
    """

    def __init__(self, dataset_path: str, scale: int = 2,
                 degradation: str = "bicubic", n_images: int = None) -> None:
        """
        Constructor method of the class

        :param dataset_path: path of the folder containing dataset images (str)
        :param scales: list containing the resolution scales to consider (list, default None)
        :param degradation: type of degraded images to use (str, default "bicubic")
        :param patch_size: size of the square (patch_size x patch_size) lr patches to extract (int, default 64)
        """

        super(ValidationDataset, self).__init__()

        # define dataset path
        self.dataset_path = dataset_path

        # define degradation method to use
        self.degradation = degradation.lower()

        # extract the image file names from the dataset
        self.filenames = sorted(os.listdir(os.path.join(dataset_path, "hr")))
        if n_images:
            # sample n_images samples from the validation set in order to use less images for the validation
            self.filenames = random.choices(self.filenames, k=n_images)

        # define scale
        self.scale = scale

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

        # extract the LR image for the current scale from the LR folder
        lr_image_path = os.path.join(self.dataset_path, "lr", self.degradation, "x" + str(self.scale),
                                     file_name)
        lr_image = io.imread(lr_image_path)

        # define the output tuple as empty
        output_tuple = (self.scale, T.ToTensor()(lr_image.copy()), T.ToTensor()(hr_image.copy()))

        return output_tuple


class TestDataset(data.Dataset):
    """
    PyTorch dataset loading a test dataset of hr images and creating lr images by degrading
    """

    def __init__(self, dataset_path: str, scale: int = 3, degradation: str = "bicubic") -> None:
        """
        Constructor method of the class

        :param dataset_path: path of the folder containing dataset images (str)
        :param scales: list containing the resolution scales to consider (list, default None)
        :param degradation: type of degraded images to use (str, default "bicubic")
        """

        super(TestDataset, self).__init__()

        # define dataset path
        self.dataset_path = dataset_path

        # define degradation method to use
        self.degradation = degradation.lower()

        # define scales to use
        self.scale = scale

        # extract the image file names from the dataset
        self.filenames = sorted(os.listdir(dataset_path))

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
        hr_image_path = os.path.join(self.dataset_path, file_name)
        hr_image = io.imread(hr_image_path)

        # if image is grayscale, add the third dimension (channel)
        if len(hr_image.shape) < 3:
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_GRAY2RGB)

        lr_image, hr_image = self.degrade(hr_image)

        # define the output tuple and return it
        output_tuple = (self.scale, T.ToTensor()(lr_image.copy()), T.ToTensor()(hr_image.copy()))

        return output_tuple

    def degrade(self, image: np.ndarray):
        # compute the output size of the degraded image
        height = image.shape[0]
        width = image.shape[1]

        # compute the actual rescaled size of the degraded image
        downscaled_height = height / self.scale
        downscaled_width = width / self.scale

        # if either the downscale width or height is a float number (meaning the original height/width is not divisible)
        # by the scale
        if not downscaled_width.is_integer() or not downscaled_height.is_integer():
            # compute the actual rounded downscaled sizes
            height = height // self.scale
            width = width // self.scale

            # upscale the rounded sizes by using the scale to compute the SR resolution
            output_height = height * self.scale
            output_width = width * self.scale

            # rescale the hr image to the SR output size
            image = cv2.resize(image, dsize=(output_width, output_height), interpolation=cv2.INTER_CUBIC)
        else:
            # just compute the actual rounded height
            height = height // self.scale
            width = width // self.scale

        # apply the degradation module
        if self.degradation == "bicubic":
            # if bicubic, just apply a bicubic downsampling to the image
            degradated_image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        elif self.degradation == "blur_down":
            # if blur down, first apply a gaussian blur with 7x7 kernel and sigma 1.6 to the image
            degradated_image = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=1.6)

            # then, resize the image with bicubic downsampling
            degradated_image = cv2.resize(degradated_image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        elif self.degradation == "down_noise":
            # if down noise, first resize the image with bicubic downsampling
            degradated_image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

            # get the height and width of the image
            height = degradated_image.shape[0]
            width = degradated_image.shape[1]

            # compute the standard deviation of the image to use it as sigma for the noise
            sigma = degradated_image.std()

            # generate a gaussian noise array of the size of the image and with variance equal to the std of the image
            gauss = np.random.normal(0, sigma, (height, width))

            # reduce the noise to 30 % of the total noise
            percentage = 0.3
            gauss *= percentage

            # add noise to all the channels of the image
            noisy = degradated_image + np.stack((gauss,) * 3, axis=-1)

            # clip noisy image min and max to 0 and 255
            noisy = np.clip(noisy, 0, 255)

            # convert the noisy image to int
            degradated_image = np.uint8(noisy)
        else:
            degradated_image = image

        return degradated_image, image
