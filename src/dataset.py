from builtins import tuple
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import os
import numpy as np
import random
from tqdm.auto import tqdm
import h5py
import time
import zipfile


class PatchesDatasetH5(data.Dataset):
    """
    PyTorch dataset loading DIV2K HR and LR images from a h5 file
    """

    def __init__(self, zip_file_path: str, image_format: str = "png", scales: list = None, split: str = "train",
                 degradation: str = "bicubic", patch_size: int = 64, augment: bool = True) -> None:
        """
        TODO
        """

        super(PatchesDatasetH5, self).__init__()

        # define scales to use
        if not scales:
            self.scales = [2, 3, 4]
        else:
            self.scales = scales

        scales_check = [f"x{scale}" for scale in self.scales]

        # define degradation method to use
        self.degradation = degradation.lower()

        # define patch size
        self.patch_size = patch_size

        # define split
        self.split = split.lower()

        # define to tensor function
        self.to_tensor = T.ToTensor()

        # define augmentation
        self.augment = augment

        image_format = image_format.lower()

        # get parent path
        parent_path = os.path.dirname(os.path.abspath(zip_file_path))

        # get zip file name
        zip_file_name = os.path.splitext(os.path.basename(os.path.abspath(zip_file_path)))[0]

        # define the hdf5 folder path
        hdf5_folder = os.path.join(parent_path, zip_file_name)

        # if folder does not exist
        if not os.path.exists(os.path.join(parent_path, zip_file_name)):
            filenames = []
        else:
            # extract file name list
            _, _, filenames = next(os.walk(hdf5_folder))

        # if hr file is missing for the chosen split, create it
        if not any('_hr' in filename and split in filename for filename in filenames):
            create_div2k_h5_file(zip_file_path, split=self.split, quality="hr", image_format=image_format)

        # create the missing lr scaled files
        for scale in scales_check:
            if not any(split in filename and scale in filename and self.degradation in filename for filename in
                       filenames):
                create_div2k_h5_file(zip_file_path, split=self.split, quality="lr", degradation=self.degradation,
                                     scale=scale, image_format=image_format)

        # set the dataset folder path
        self.dataset_path = Path(hdf5_folder)

        # open the hr hdf5 file
        self.hr_file = h5py.File(os.path.join(self.dataset_path, f"{self.split}_hr.hdf5"), 'r')

        # open the lr hdf5 files
        self.lr_files = {
            scale: h5py.File(os.path.join(self.dataset_path, f"{self.split}_lr_{self.degradation}_x{scale}.hdf5"), 'r')
            for scale in self.scales}

        # check if the files contain the same number of images
        if any(len(file.keys()) != len(self.hr_file.keys()) for _, file in self.lr_files.items()):
            raise Exception("HR and LR files does not contain the same number of images.")

        # define file_names
        self.file_list = list(self.hr_file.keys())

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: length of the dataset
        """

        return len(self.file_list)

    def __getitem__(self, item) -> tuple:
        """
        TODO
        """

        # select file
        file = self.file_list[item]

        # extract hr image
        hr = np.asarray(self.hr_file[file])

        output_tuple = ()

        # extract the lr images
        for scale, lr_file in self.lr_files.items():
            # convert image to PyTorch tensor
            lr = np.asarray(lr_file[f'{file}x{scale}'])

            # extract the lr and hr patches
            lr_patch, hr_patch = random_crop(lr, hr, scale)

            # if augmentation is required
            if self.augment:
                # flip the patches
                lr_patch, hr_patch = random_horizontal_flip(lr_patch, hr_patch)

                # rotate the patches
                lr_patch, hr_patch = random_90_rotation(lr_patch, hr_patch)

            output_tuple += (scale, self.to_tensor(lr_patch), self.to_tensor(hr_patch))

        return output_tuple


def create_div2k_h5_file(zip_file_path: str, split: str = "train", degradation: str = "bicubic",
                         scale: str = "x2", quality: str = "lr", image_format: str = "png") -> None:
    """
    TODO
    """

    if quality.lower() == "lr":
        print(f"\nCreating {quality} {degradation} {scale} hdf5 file (it may take a while)...")
    else:
        print(f"\nCreating {quality} hdf5 file (it may take a while)...")

    # open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:

        # get parent path
        parent_path = os.path.dirname(os.path.abspath(zip_file.filename))

        # get zip file name
        zip_file_name = os.path.splitext(os.path.basename(os.path.abspath(zip_file.filename)))[0]

        # if hdf5 directory does not exists, create it
        if not os.path.exists(os.path.join(parent_path, zip_file_name)):
            os.makedirs(os.path.join(parent_path, zip_file_name))
            print("Hdf5 directory created.")

        # get all the files in the zip
        files_and_folders = zip_file.namelist()

        # filter the filenames using the specified parameters
        extracted_files = [file for file in files_and_folders if
                           split.lower() in file and
                           quality.lower() in file and
                           f".{image_format.lower()}" in file]

        # if the quality is lr, filter by also using degradation and scale
        if quality.lower() == "lr":
            extracted_files = [file for file in extracted_files if
                               degradation.lower() in file and
                               scale.lower() in file]

        # sort extracted file names
        extracted_files.sort()

        # define output file name
        if quality.lower() == "lr":
            output_file_name = f"{split}_{quality}_{degradation}_{scale}.hdf5"
        else:
            output_file_name = f"{split}_{quality}.hdf5"

        # open the output file
        with h5py.File(os.path.join(os.path.join(parent_path, zip_file_name, output_file_name)),
                       "w") as output_file:

            # for each image file extracted
            for file in tqdm(extracted_files, total=len(extracted_files)):
                img_file = zip_file.open(file)
                image = Image.open(img_file)

                # convert image to numpy array
                img = np.asarray(image)

                # close image and delete to free space
                image.close()
                del image

                # extract the name of the file from the path
                image_file_name = os.path.splitext(os.path.basename(file))[0].lower()

                # save the image on the hdf5 dataset
                output_file.create_dataset(image_file_name, data=img)

    print("Done!")


def collate_fn(batch: list) -> tuple:
    """
    TODO
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


ds = PatchesDatasetH5("../data/div2k.zip")
dload = data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2,
                        pin_memory=True)

start = time.time()
for scale, lr, hr in tqdm(dload):
    print(scale, lr.size(), hr.size())
end = time.time()

print(end - start)
