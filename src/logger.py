from abc import ABC, abstractmethod
import wandb
from src.utils import random_string


class WandbLogger:
    """
    Class that describes a Wandb logger
    """

    def __init__(self,
                 name: str,
                 config: dict,
                 project: str = "ML4CV_project",
                 entity: str = "pierclgr") -> None:
        """
        Constructor method of a Wandb logger

        :param name: the name of the Wandb run in which we're logging (str)
        :param config: the configuration of the current run to be logged (dict)
        :param project: the name of the Wandb project on which to log, default value is "AAS_project" (str)
        :param entity: the name of the entity on Wandb on which to log, default value is "pierclgr" (str)

        :return: None
        """

        self.name = name + "_" + random_string()
        self.project = project
        self.entity = entity
        self.config = config

        self.logger = wandb.init(project=self.project, name=self.name, entity=self.entity, config=self.config)

    def finish(self) -> None:
        """
        Method closes and finishes the Wandb logger

        :return: None
        """
        self.logger.finish()

    def log(self, metric: str, value: float, step: int) -> None:
        """
        Method that logs the metrics to the define run

        :param metric: the metric we're logging (str)
        :param value: the value of the metric we are logging at this time (float)
        :param step: the current time at which we're logging the metric value (int)

        :return: None
        """

        # log the input metric to the Wandb run
        self.logger.log({metric: value}, step=step)
