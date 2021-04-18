import os
import torch

# import torch_xla library if runtime is using a Colab TPU
if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm


def get_device() -> str:
    """
    Get the current machine device to use

    :returns: the current machine device
    """

    # if the current runtime is using a Colab TPU, define a flag specifying that TPU will be used
    if 'COLAB_TPU_ADDR' in os.environ:
        use_tpu = True
    else:
        use_tpu = False

    # if TPU is available, use it as device
    if use_tpu:
        device = xm.xla_device()
    else:
        # otherwise use CUDA device or CPU accordingly to the one available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> Using {device} device")

    return device

get_device()