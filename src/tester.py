import importlib
import os
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils import data
import numpy as np
from src.datasets import TestDataset
from src.logger import WandbLogger
from src.utils import get_device, set_seeds, count_parameters
from tqdm.auto import tqdm
from src.metrics import compute_metrics
import hydra
import pprint


class Tester:
    def __init__(self, config):
        self.config = config

        # create the model
        self.model = getattr(importlib.import_module("src.models"), config.model)(
            input_channels=config.image_channels)

        # get the device
        self.device = get_device()
        self.model = self.model.to(self.device)

        # configure logger
        configuration = OmegaConf.to_object(config)
        pp = pprint.PrettyPrinter()
        pp.pprint(configuration)
        if config.wandb.logging:
            self.logger = WandbLogger(name=config.wandb.run_name, config=configuration,
                                      project=config.wandb.project_name, entity=config.wandb.entity_name)
        else:
            self.logger = None

        # create test dataloader from the given testing dataset
        test_dataset = TestDataset(config.test_dataset.path,
                                   scale=config.test_dataset.scale,
                                   degradation=config.test_dataset.degradation)
        self.test_dataloader = data.DataLoader(test_dataset,
                                               batch_size=config.test_dataset.batch_size,
                                               shuffle=config.test_dataset.shuffle,
                                               num_workers=config.test_dataset.num_workers,
                                               pin_memory=config.test_dataset.pin_memory)

        # load the weights from the saved file
        self.load(self.config.output_model_file)

    def test(self):
        print("Testing...")

        # set model to eval mode
        self.model.eval()

        # initialize testing metrics
        test_psnr = 0
        test_ssim = 0
        test_samples = 0
        test_sr_hr_comparisons = []

        # disable gradient computation
        with torch.no_grad():
            for scale, lr, hr in tqdm(self.test_dataloader, position=0):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                batch_size = lr.size()[0]

                # do forward step in the model to compute sr images
                sr = self.model(lr, scale)

                # convert the two image batches to numpy array and reshape to have channels in last dimension
                hr = hr.cpu().detach().numpy().transpose(0, 2, 3, 1)
                sr = sr.cpu().detach().numpy().transpose(0, 2, 3, 1)

                # comupute psnr and ssim for the current testing batch
                psnr, ssim = compute_metrics(hr, sr)

                # add metrics of the current batch to the total sum
                test_samples += batch_size
                test_psnr += np.sum(psnr)
                test_ssim += np.sum(ssim)

                # create an image containing the sr and hr image side by side and append to the array of comparison
                # images
                sr_hr = np.concatenate((sr[0], hr[0]), axis=1)
                test_sr_hr_comparisons.append(sr_hr)

            # compute the average metrics value for the dataset
            test_psnr = round(test_psnr / test_samples, 2)
            test_ssim = round(test_ssim / test_samples, 4)

            # log the average psnr and ssim of the dataset and the images
            if self.logger:
                self.logger.log("test_psnr", test_psnr, summary=True)
                self.logger.log("test_ssim", test_ssim, summary=True)
                self.logger.log_images(test_sr_hr_comparisons[:self.config.wandb.n_images_to_log],
                                       caption="Left: SR, Right: ground truth (HR)",
                                       name="Testing samples", step=0)

            # print the metrics at the end of the epoch
            print("Samples:", test_samples,
                  "\n\t- test psnr:", test_psnr,
                  "\n\t- test ssim:", test_ssim)

        return test_psnr, test_ssim, test_sr_hr_comparisons

    def load(self, filename: str) -> None:
        filename = f"{filename}.pt"
        trained_model_path = self.config.model_folder
        if os.path.isdir(trained_model_path):
            file_path = f"{trained_model_path}{filename}"
            if os.path.isfile(file_path):
                print(f"Loading model from {file_path}...")
                weights = torch.load(file_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(weights)
                print("Done!")
            else:
                print("The specified file does not exist in the trained models directory.")
        else:
            print("The directory of the trained models does not exist.")


@hydra.main(version_base=None, config_path="../config/", config_name="testing")
def main(config: DictConfig):
    # set seeds for reproducibility
    set_seeds(config.seed)

    # create tester with the given testing configuration
    tester = Tester(config)
    count_parameters(tester.model)

    # run the test
    tester.test()

    # if logging is enabled, finish the logger
    if config.wandb.logging:
        tester.logger.finish()


if __name__ == "__main__":
    main()
