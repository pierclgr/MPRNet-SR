import os
import torch


# function to get the current available device (CPU, GPU or TPU)
def get_device() -> torch.device:
    """
    Get the current machine device to use

    :returns: the current machine device
    """

    # use CUDA device or CPU accordingly to the one available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if the device is a GPU
    if torch.cuda.is_available():
        # print the details of the given GPU
        stream = os.popen('nvidia-smi')
        output = stream.read()
        print(output)

    print(f">>> Using {device} device")

    return device


def rename() -> None:
    dir = os.getcwd() + "/validation/lr/unknown/x2/"

    for filename in sorted(os.listdir(dir)):
        name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        name = name[0:4]

        new_name = dir + name + ext
        old_name = dir + filename

        os.rename(old_name, new_name)
