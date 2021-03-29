import torch.utils.data as data
from PIL import Image
import glob
import torch
import torchvision.transforms as T
from pathlib import Path


class DIV2K(data.Dataset):
    """
    PyTorch dataset loading images from a folder
    """

    def __init__(self, path: str, image_format: str):
        """
        Constructor of the class

        :param path: path of the folder containing the data
        :param image_format: format of the image files in the folder
        """

        # set folder path
        self.folder_path = Path(path)

        # add images to the dataset object as list of file paths
        self.image_files = list(self.folder_path.glob(f"*.{image_format}"))

        # define list of scales
        self.scales = [2, 3, 4]


    def __getitem__(self, item):
        """
        Return an image from the dataset by applying the transformation pipeline

        :param item: the image to return
        :return: selected image as a PyTorch tensor
        """

        # select file path
        file_path = self.image_files[item]

        # open the file
        with open(file_path, "rb") as image_file:
            # open current image file as a PIL Image
            im = Image.open(image_file)

            # return image as PyTorch tensor
            return T.ToTensor(im)


d = DIV2K("../data/div2k/train", "png")
