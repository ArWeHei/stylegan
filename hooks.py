import tensorflow as tf
import numpy as np

import os
import time
import sys

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger

from tensorboardX import SummaryWriter

class MarginPlottingHook(Hook):
    def __init__(
        self,
        scalars={},
        interval=100,
        root_path="logs",
        summary_writer=None
    ):
        """Args:
            scalars (dict): Scalar ops.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        """


        self.scalars = scalars
        self.keys = list(scalars.keys())
        self.interval = interval

        self.root = root_path
        self.logger = get_logger(self)
        if summary_writer is None:
            self.tb_logger = SummaryWriter(root_path)
        else:
            self.tb_logger = summary_writer

        self.results_log = {key:[] for key in self.keys()}

        self.tb_logger.add_custom_scalars_marginchart([key+'_mean', key+'+std', key+'-std'])


    def before_step(self, batch_index, fetches, feeds, batch):
        fetches["custom_scalars"] = self.scalars


    def after_step(self, batch_index, last_results):
        step = last_results["global_step"]
        last_results = last_results["custom_scalars"]
        pprint(last_results)
        for (key, value) in last_results.items():
            self.results_log[key] += value

        if batch_index % self.interval == 0:
            for name in sorted(self.keys()):
                m = np.mean(self.results_log[key])
                s = np.std(self.results_log[key])
                self.tb_logger.add_scalar(key+'_mean', m, step)
                self.tb_logger.add_scalar(key+'+std', m+s, step)
                self.tb_logger.add_scalar(key+'-std', m-s, step)
                self.logger.info(f"{name}: {m} ± {s}")


class ImageLoggingHook(Hook):
    def __init__(
        self,
        images={},
        interval=100,
        root_path="logs",
    ):

        self.images = images
        self.keys = list(images.keys())
        self.interval = interval

        self.root = root_path
        self.logger = get_logger(self)
        if summary_writer is None:
            self.tb_logger = SummaryWriter(root_path)
        else:
            self.tb_logger = summary_writer


    def before_step(self, batch_index, fetches, feeds, batch):
        if batch_index % self.interval == 0:
            fetches["images"] = self.images


    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results["global_step"]
            last_results = last_results["images"]
            for (key, value) in last_results.items():
                self.tb_logger.add_image(key, value, step)

#class LODHook(Hook):

