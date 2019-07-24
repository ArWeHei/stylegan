import tensorflow as tf
import numpy as np

import os
import time
import sys

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger

from tensorboardX import SummaryWriter

class CustomTFScalarLoggingHook(Hook):
    """Supply and evaluate logging ops at an intervall of training steps."""
    # do this differently:
    #   start of with ops a.k.a. tensors that are added to the feed dict
    #   after the run the result is retrieved and processed, i.e. square the value, save number of results etc.
    #   save the result in an attribute
    #   when interval is reached, write down the data using tensorboardX

    def __init__(
        self,
        scalars={},
        interval=100,
        root_path="logs",
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
        self.tb_logger = SummaryWriter(root_path)

        #self.fetch_dict = {"custom_scalars": scalars}
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


#class LODHook(Hook):

