import importlib
import os
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from torch.utils import data
from src.datasets import TrainDataset, ValidationDataset
from src.logger import WandbLogger
from src.metrics import compute_metrics
from src.utils import get_device, set_seeds
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_


class Trainer:
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.optimizer.learning_rate

        # create the model
        self.model = getattr(importlib.import_module("src.models"), config.model)(
            input_channels=config.image_channels, n_features=config.n_features)

        # define the loss
        self.criterion = getattr(importlib.import_module("torch.nn"), config.loss)()

        # define the optimizer
        self.optimizer = getattr(importlib.import_module("adamp"), config.optimizer.name)(
            params=self.model.parameters(),
            betas=tuple(config.optimizer.betas),
            eps=config.optimizer.eps,
            lr=self.learning_rate
        )

        # get the device
        self.device = get_device()
        self.model = self.model.to(self.device)

        # configure logger
        configuration = OmegaConf.to_object(config)
        if config.wandb.logging:
            self.logger = WandbLogger(name=config.wandb.run_name, config=configuration,
                                      project=config.wandb.project_name, entity=config.wandb.entity_name)
        else:
            self.logger = None

        # create train dataloader from the given training dataset
        train_dataset = TrainDataset(config.train_dataset.path,
                                     scales=list(config.train_dataset.scales),
                                     degradation=config.train_dataset.degradation,
                                     patch_size=config.train_dataset.patch_size,
                                     augment=config.train_dataset.augment)
        self.train_dataloader = data.DataLoader(train_dataset,
                                                batch_size=config.train_dataset.batch_size,
                                                shuffle=config.train_dataset.shuffle,
                                                collate_fn=TrainDataset.collate_fn,
                                                num_workers=config.train_dataset.num_workers,
                                                pin_memory=config.train_dataset.pin_memory)

        # create validation dataloader from the given validation dataset
        val_dataset = ValidationDataset(config.val_dataset.path,
                                        scale=config.val_dataset.scale,
                                        degradation=config.val_dataset.degradation,
                                        n_images=config.val_dataset.n_images_to_use)
        self.val_dataloader = data.DataLoader(val_dataset,
                                              batch_size=config.val_dataset.batch_size,
                                              shuffle=config.val_dataset.shuffle,
                                              num_workers=config.val_dataset.num_workers,
                                              pin_memory=config.val_dataset.pin_memory)

    def train(self):

        # set initial values for total training epochs and steps
        print("Starting training...")
        steps = 0
        epochs = 0
        finished = False

        steps_pbar = tqdm(total=self.config.max_training_steps, position=0)

        # while the training is not finished (i.e. we haven't reached the max number of training steps)
        while not finished:

            # set the model in training mode since at the end of each epoch the model is set to eval mode by the eval
            # method
            self.model.train()

            # initialize the current epoch metrics
            train_loss = 0
            train_samples = 0
            train_psnr = 0
            train_ssim = 0
            best_train_psnr = 0
            best_train_ssim = 0
            best_val_psnr = 0
            best_val_ssim = 0
            train_sr_hr_comparisons = []

            # for each batch in the training set
            for scale, lrs, hrs in tqdm(self.train_dataloader, position=0):

                # send lr and hr batches to device
                lrs = lrs.to(self.device)
                hrs = hrs.to(self.device)
                batch_size = lrs.size()[0]

                # zero the gradients
                self.optimizer.zero_grad()

                # do forward step in the model to compute sr images
                srs = self.model(lrs, scale)

                # compute loss between srs images and hrs
                loss = self.criterion(srs, hrs)

                # add current loss to the training loss
                train_loss += loss.item() * batch_size
                train_samples += batch_size

                # convert the two image batches to numpy array and reshape to have channels in last dimension
                hrs = hrs.cpu().detach().numpy().transpose(0, 2, 3, 1)
                srs = srs.cpu().detach().numpy().transpose(0, 2, 3, 1)

                # compute the current training metrics
                psnr, ssim = compute_metrics(hrs, srs)

                # add metrics of the current batch to the total sum
                train_psnr += np.sum(psnr)
                train_ssim += np.sum(ssim)

                # create an image containing the sr and hr image side by side and append to the array of comparison
                # images
                sr_hr = np.concatenate((srs[0], hrs[0]), axis=1)
                train_sr_hr_comparisons.append(sr_hr)

                # do a gradient descent step
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

                # increment the number of total steps
                steps += 1
                steps_pbar.update(1)

                # half learning rate
                if (steps % self.config.optimizer.halving_steps) == 0:
                    self.learning_rate /= 2
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                # if number of maximum training steps is reached
                if steps >= self.config.max_training_steps:
                    # finish the training by breaking the for loop and the outer loop
                    finished = True
                    break

            # compute the current epoch training loss
            train_loss /= train_samples

            # compute the average metrics for the current training epoch
            train_psnr = round(train_psnr / train_samples, 2)
            train_ssim = round(train_ssim / train_samples, 4)

            # evaluate the model for each scale at the end of the epoch (when we looped the entire training set) and get
            # the validation loss and metrics
            val_loss, val_psnr, val_ssim, val_sr_hr_comparisons = self.validate()

            # compute the new best train metrics
            if train_psnr > best_train_psnr:
                best_train_psnr = train_psnr
            if train_ssim > best_train_ssim:
                best_train_ssim = train_ssim

            # compute the new best validation metrics
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim

            # print the metrics at the end of the epoch
            print("Epoch:", epochs + 1, "- total_steps:", steps + 1,
                  "\n\tTRAIN",
                  "\n\t- train loss:", train_loss,
                  "\n\t- train psnr:", train_psnr,
                  "\n\t- best train psnr:", best_train_psnr,
                  "\n\t- train ssim:", train_ssim,
                  "\n\t- best train ssim:", best_train_ssim,
                  "\n\tVAL",
                  "\n\t- val loss:", val_loss,
                  "\n\t- val psnr:", val_psnr,
                  "\n\t- best val psnr:", best_val_psnr,
                  "\n\t- val ssim:", val_ssim,
                  "\n\t- best val ssim:", best_val_ssim)

            # log metrics to the logger at each training step if required
            if self.logger:
                self.logger.log("train_loss", train_loss, epochs)
                self.logger.log("train_psnr", train_psnr, epochs)
                self.logger.log("train_ssim", train_ssim, epochs)
                self.logger.log("best_train_psnr", best_train_psnr, summary=True)
                self.logger.log("best_train_ssim", best_train_ssim, summary=True)
                self.logger.log("val_loss", val_loss, epochs)
                self.logger.log("val_psnr", val_psnr, epochs)
                self.logger.log("val_ssim", val_ssim, epochs)
                self.logger.log("best_val_psnr", best_val_psnr, summary=True)
                self.logger.log("best_val_ssim", best_val_ssim, summary=True)
                self.logger.log_images(train_sr_hr_comparisons[:self.config.wandb.n_images_to_log],
                                       caption="Left: SR, Right: ground truth (HR)",
                                       name="Training samples", step=epochs)
                self.logger.log_images(val_sr_hr_comparisons[:self.config.wandb.n_images_to_log],
                                       caption="Left: SR, Right: ground truth (HR)",
                                       name="Validation samples", step=epochs)

            # increment number of epochs
            epochs += 1

        print("Training finished! Saving model...")
        self.save(self.config.output_model_file)
        print("Done!")
        steps_pbar.close()

    def validate(self):
        print("Evaluating...")

        # set model to eval mode
        self.model.eval()

        # initialize current validation epoch metrics
        val_samples = 0
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        val_sr_hr_comparisons = []

        # disable gradient computation
        with torch.no_grad():
            for scale, lr, hr in tqdm(self.val_dataloader, position=0):
                # send lr and hr to device
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                batch_size = lr.size()[0]

                # do forward step in the model to compute sr images
                sr = self.model(lr, scale)

                # compute the validation loss for the current scale
                loss = self.criterion(sr, hr)
                val_loss += loss.item() * batch_size
                val_samples += batch_size

                # convert the two image batches to numpy array and reshape to have channels in last dimension
                hr = hr.cpu().detach().numpy().transpose(0, 2, 3, 1)
                sr = sr.cpu().detach().numpy().transpose(0, 2, 3, 1)

                # comupute psnr and ssim for the current validation sample
                psnr, ssim = compute_metrics(hr, sr)

                # add metrics of the current batch to the total sum
                val_psnr += np.sum(psnr)
                val_ssim += np.sum(ssim)

                # create an image containing the sr and hr image side by side and append to the array of comparison
                # images
                sr_hr = np.concatenate((sr[0], hr[0]), axis=1)
                val_sr_hr_comparisons.append(sr_hr)

            # compute the average val loss for the current validation epoch
            val_loss /= val_samples

            # compute the average metrics for the current validation epoch
            val_psnr = round(val_psnr / val_samples, 2)
            val_ssim = round(val_ssim / val_samples, 4)

        return val_loss, val_psnr, val_ssim, val_sr_hr_comparisons

    def save(self, filename: str):
        filename = f"{filename}.pt"
        trained_model_path = self.config.model_folder
        if not os.path.isdir(trained_model_path):
            os.makedirs(trained_model_path)
        file_path = f"{trained_model_path}{filename}.pt"

        print(f"Saving trained model to {filename}.pt...")

        # save network weights
        checkpoint = {"model_weights", self.model.state_dict()}
        torch.save(checkpoint, file_path)

    def load(self, filename: str) -> None:
        filename = f"{filename}.pt"
        trained_model_path = self.config.model_folder
        if os.path.isdir(trained_model_path):
            file_path = f"{trained_model_path}{filename}"
            if os.path.isfile(file_path):
                print(f"Loading model from {filename}...")
                checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(checkpoint['model_weights'])
            else:
                print("The specified file does not exist in the trained models directory.")
        else:
            print("The directory of the trained models does not exist.")


@hydra.main(version_base=None, config_path="../config/", config_name="training")
def main(config: DictConfig):
    # set seeds for reproducibility
    set_seeds(config.seed)

    # create trainer with the given testing configuration
    trainer = Trainer(config)

    from src.utils import count_parameters
    print(count_parameters(trainer.model))

    # run the training
    trainer.train()

    # if logging is enabled, finish the logger
    if config.wandb.logging:
        trainer.logger.finish()


if __name__ == "__main__":
    main()
