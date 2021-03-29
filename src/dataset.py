from builtins import tuple
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import os
import numpy as np
import random


class PatchesDataset(data.Dataset):
    """
    PyTorch dataset loading DIV2K HR and LR images from a folder
    """

    def __init__(self, path: str, image_format: str = "png", scales: list = None, degradation: str = "bicubic",
                 patch_size: int = 64, augment: bool = True) -> None:
        """
        Constructor of the class

        :param path: path of the dataset folder (str)
        :param image_format: format of the image files (str, default "png")
        :param scales: list of scales to use (list, default None)
        :param degradation: degradation to use (str, default "bicubic")
        :param patch_size: size of the patches (int, default 64)
        :param augment: flag to control the augmentation of the data (bool, default True)
        """

        # set dataset folder path
        self.dataset_path = Path(path)

        # set HR folder path
        self.hr_path = self.dataset_path / "hr"

        # set LR folder path
        self.lr_path = self.dataset_path / "lr"

        # extract file names
        file_list = list(self.hr_path.glob(f"*.{image_format}"))
        self.file_list = [os.path.basename(file) for file in file_list]

        # define scales to use
        if not scales:
            self.scales = [2, 3, 4]
        else:
            self.scales = scales

        # define degradation method to use
        self.degradation = degradation

        # define patch size
        self.patch_size = patch_size

        # define to tensor function
        self.to_tensor = T.ToTensor()

        # define augmentation
        self.augment = augment

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: length of the dataset
        """

        return len(self.file_list)

    def __getitem__(self, item) -> tuple:
        """
        Method to get a patch from an image in the dataset. It returns a tuple containing a tuple for each scale.
        Each scale tuple contains, in order:
            - the scale
            - the lr patch as PyTorch tensor
            - the hr patch as PyTorch tensor

        :param item: index of the item to get
        :return: tuple containing hr and lr patches couples for each scale
        """

        # select file
        file = self.file_list[item]

        # separate file name and extension
        file_name, file_extension = os.path.splitext(file)

        # extract hr image path
        hr_image_path = self.hr_path / file

        # open the hr file
        with open(hr_image_path, "rb") as image_file:
            # open hr image file as a PIL Image
            hr_im = Image.open(image_file)
            hr_im = hr_im.convert("RGB")

            # convert image to numpy array and save
            hr = np.asarray(hr_im)

            # close image and delete to free space
            hr_im.close()
            del hr_im

        # create the dictionary containing lr file names for each scale
        lr_file_names = {scale: f"{file_name}x{scale}{file_extension}" for scale in self.scales}

        # create the dictionary containing the lr images paths for each scale
        lr_image_paths = {scale: self.lr_path / self.degradation / f"x{scale}" / file_name for scale, file_name in
                          lr_file_names.items()}

        # create dictionary containing lr images for each scale
        lr_images = {}

        # extract the lr images
        for scale, lr_image_path in lr_image_paths.items():
            with open(lr_image_path, "rb") as image_file:
                # open lr image file as PIL image
                lr_im = Image.open(image_file)
                lr_im = lr_im.convert("RGB")

                # convert image to PyTorch tensor
                lr = np.asarray(lr_im)

                # close image
                lr_im.close()
                del lr_im

            # add current lr image to the dictionary
            lr_images[scale] = lr

        # extract the lr-hr patches for each scale
        lr_hr_patches = {scale: random_crop(cur_lr, hr, scale) for scale, cur_lr in lr_images.items()}

        # if agumentation is required
        if self.augment:
            # flip the patches
            lr_hr_patches = {scale: random_horizontal_flip(lr_patch, hr_patch) for scale, (lr_patch, hr_patch) in
                             lr_hr_patches.items()}

            # rotate the patches
            lr_hr_patches = {scale: random_90_rotation(lr_patch, hr_patch) for scale, (lr_patch, hr_patch) in
                             lr_hr_patches.items()}

        # extract and return lr-hr patches pairs for each scale as PyTorch tensors
        return tuple((scale, self.to_tensor(lr), self.to_tensor(hr)) for scale, (lr, hr) in lr_hr_patches.items())


def random_crop(lr: np.ndarray, hr: np.ndarray, scale: int = 2, patch_size: int = 64) -> tuple:
    """
    Extracts a random patch of the given size from an input lr image and the corresponding hr patch scaled using the
    given scale.

    :param lr: low-resolution image to extract the patch from (ndarray)
    :param hr: high-resolution image to extract the patch from (ndarray)
    :param scale: scale to use for the extraction of the hr patch (int, default 2)
    :param patch_size: size of the low resolution (square) patch (int, default 64)
    :return: tuple containing the extacted lr and hr patches
    """

    # extract size of the lr image
    height, width = lr.shape[:-1]

    # extract random starting coordinates of the patch in the lr image
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)

    # compute the starting coordinates of the patch in the hr image
    hr_patch_size = patch_size * scale
    hx, hy = x * scale, y * scale

    # extract the patch from the two images
    lr = lr[y:y + patch_size, x:x + patch_size].copy()
    hr = hr[hy:hy + hr_patch_size, hx:hx + hr_patch_size].copy()

    return lr, hr


def random_horizontal_flip(lr: np.ndarray, hr: np.ndarray, p: float = .5) -> tuple:
    """
    Randomly applies horizontal flip to the given lr and hr patches with the given flipping probability

    :param lr: low-resolution patch to flip (ndarray)
    :param hr: high-resolution patch to flip (ndarray)
    :param p: probability of the flipping (float, default 0.5)
    :return: tuple containing the flipped (or not) lr and hr patches
    """

    # flip horizontally the images
    if random.random() < p:
        lr = np.fliplr(lr)
        hr = np.fliplr(hr)

    return lr.copy(), hr.copy()


def random_90_rotation(lr: np.ndarray, hr: np.ndarray) -> tuple:
    """
    Randomly applies a 90° rotation (or not) to the given lr and hr patches

    :param lr: low-resolution patch to rotate (ndarray)
    :param hr: high-resolution patch to rotate (ndarray)
    :return: tuple containing the rotated (or not) lr and hr patches
    """

    # choose a rotation angle (0, 90, -90)
    n_rotations = random.choice([0, 1, 3])

    # rotate the images
    lr = np.rot90(lr, n_rotations)
    hr = np.rot90(hr, n_rotations)

    return lr.copy(), hr.copy()


def create_patches_batch(batch: list) -> tuple:
    """
    Collate function for the patches dataset to create a batch of patches. Selects randomly a scale between the ones
    used in the dataset and extracts the corresponding scaled batch.

    :param batch: batch containing the items extracted from the used dataset (list)
    :return: tuple containing the scale, the lr and the hr patches batches
    """

    # select randomly a scale and it's batch based on the scales contained in the dataset
    scaled_batch = random.choice(list(zip(*batch)))

    # unzip the scaled batch
    scale, lr, hr = zip(*scaled_batch)

    # set the scale
    scale = scale[0]

    # stack the tensor images to a unique batch tensor
    lr = torch.stack(lr)
    hr = torch.stack(hr)

    return scale, lr, hr


d = PatchesDataset("../data/div2k/train")
dload = data.DataLoader(d, batch_size=3, shuffle=False, collate_fn=create_patches_batch)


for scale, lr, hr in dload:
    print(scale)
    print(lr.size())

    print(hr.size())
    break
