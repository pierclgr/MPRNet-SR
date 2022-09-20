import importlib
from omegaconf import OmegaConf
from torch.utils import data
from src.datasets import TrainDataset, ValDataset
from src.logger import WandbLogger
from src.utils import get_device
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.optimizer.learning_rate

        # create the model
        self.model = getattr(importlib.import_module("src.models"), config.model)(
            input_channels=config.image_channels)

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
                                     scales=list(config.scales),
                                     degradation=config.degradation,
                                     patch_size=config.patch_size,
                                     augment=config.train_dataset.augment)
        self.train_dataloader = data.DataLoader(train_dataset,
                                                batch_size=config.train_dataset.batch_size,
                                                shuffle=config.train_dataset.shuffle,
                                                collate_fn=TrainDataset.collate_fn,
                                                num_workers=config.train_dataset.num_workers,
                                                pin_memory=config.train_dataset.pin_memory)

        # create validation dataloader from the given validation dataset
        val_dataset = TrainDataset(config.val_dataset.path,
                                   scales=list(config.scales),
                                   degradation=config.degradation,
                                   patch_size=config.patch_size,
                                   augment=config.val_dataset.augment)
        self.val_dataloader = data.DataLoader(val_dataset,
                                              batch_size=config.val_dataset.batch_size,
                                              shuffle=config.val_dataset.shuffle,
                                              collate_fn=TrainDataset.collate_fn,
                                              num_workers=config.val_dataset.num_workers,
                                              pin_memory=config.val_dataset.pin_memory)

    def train(self):

        # set initial values for total training epochs and steps
        print("Starting training...")
        steps = 0
        epochs = 0
        finished = False

        # while the training is not finished (i.e. we haven't reached the max number of training steps)
        while not finished:

            # set the model in training mode since at the end of each epoch the model is set to eval mode by the eval
            # method
            self.model.train()

            # initialize the current epoch metrics
            train_loss = 0
            train_samples = 0

            # for each batch in the training set
            for scale, lrs, hrs in tqdm(self.train_dataloader, position=0):

                # send lr and hr batches to device
                lrs = lrs.to(self.device)
                hrs = hrs.to(self.device)

                # zero the gradients
                self.optimizer.zero_grad()

                # do forward step in the model to compute sr images
                srs = self.model(lrs, scale)

                # compute loss between srs images and hrs
                loss = self.criterion(srs, hrs)

                # add current loss to the training loss
                batch_size = lrs.size()[0]
                train_loss += loss.item() * batch_size
                train_samples += batch_size

                # do a gradient descent step
                loss.backward()
                self.optimizer.step()

                # increment the number of total steps
                steps += 1

                # if number of maximum training steps is reached
                if steps >= self.config.max_training_steps:
                    # finish the training by breaking the for loop and the outer loop
                    finished = True
                    break

            # compute the current epoch training loss
            train_loss /= train_samples

            # evaluate the model for each scale at the end of the epoch (when we looped the entire training set)
            val_loss = self.validate()

            # print the metrics at the end of the epoch
            print("Epoch:", epochs + 1, "- total_steps:", steps + 1,
                  "\n\t- train loss:", train_loss,
                  "\n\t- val loss:", val_loss)

            # log metrics to the logger at each training step if required
            if self.logger:
                self.logger.log("train_loss", train_loss, epochs)
                self.logger.log("val_loss", val_loss, epochs)

            # increment number of epochs
            epochs += 1

        print("Finish!")

    def validate(self):
        print("Evaluating...")

        # set model to eval mode
        self.model.eval()

        # initialize current validation epoch metrics
        val_samples = 0
        val_loss = 0

        # disable gradient computation
        with torch.no_grad():
            for scale, lr, hr in tqdm(self.val_dataloader, position=0):
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                # do forward step in the model to compute sr images
                sr = self.model(lr, scale)

                # compute the validation loss for the current scale
                loss = self.criterion(sr, hr)

                batch_size = lr.size()[0]
                val_loss += loss.item() * batch_size
                val_samples += batch_size

            # compute the average val loss for the current validation epoch
            val_loss /= val_samples

        return val_loss

# FOR TEST FUNCTION FUTURE
# return the score


#     # get each validation sample consisting in the lr and hr images for each scale
#     for validation_sample in tqdm(self.val_dataloader, position=0):
#         # get the hr image, which is the last element of the sample, and the chuck containing lr images
#         hr = validation_sample[-1]
#         hr = hr.to(self.device)
#         lr_images = validation_sample[:-1]
#
#         # for each scale
#         for i in range(num_scales):
#             # get the current scale factor and the lr image
#             actual_tuple_index = i * 2
#             scale = lr_images[actual_tuple_index]
#             scale = scale.item()
#             lr = lr_images[actual_tuple_index + 1]
#             lr = lr.to(self.device)
#
#             # do forward step in the model to compute sr images
#             sr = self.model(lr, scale)
#
#             # compute the validation loss for the current scale
#             loss = self.criterion(sr, hr)
#
#             # add the loss to the loss corresponding to the current scale
#             val_losses[scale] += loss.item() * hr.size()[0]
#
#         # increment the number of total samples
#         val_samples += hr.size()[0]
#
#     # compute the validation losses for the current epoch
#     val_losses = {scale: loss/val_samples for scale, loss in val_losses.items()}
#
# print(val_losses)
