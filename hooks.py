import tensorflow as tf
import numpy as np

import os
import time
import sys

from math import floor, ceil

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.util import pprint, linear_var

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

        self.results_log = {key:[] for key in self.keys}
        self.prefix = 'logging/'

        layout = {'scores':{}}
        for key in self.keys:
            layout['scores'][key] = ['Margin', 
                [
                    self.prefix+key+'_mean',
                    self.prefix+key+'_p_std',
                    self.prefix+key+'_m_std',
                ]]

        self.tb_logger.add_custom_scalars(layout)

    def before_step(self, batch_index, fetches, feeds, batch):
        fetches["custom_scalars"] = self.scalars


    def after_step(self, batch_index, last_results):
        step = last_results["global_step"]
        results = last_results["custom_scalars"]
        for (key, value) in results.items():
            self.results_log[key].append(value)

        if batch_index % self.interval == 0:
            self.logger.info(f"step: {step}")
            for key in self.keys:
                m = np.mean(self.results_log[key])
                s = np.std(self.results_log[key])
                self.tb_logger.add_scalar(self.prefix+key+'_mean', m, step)
                self.tb_logger.add_scalar(self.prefix+key+'_p_std', m+s, step)
                self.tb_logger.add_scalar(self.prefix+key+'_m_std', m-s, step)
                self.logger.info(f"{key}: {m} ± {s}")
                self.results_log[key] = []


class ImageLoggingHook(Hook):
    def __init__(
        self,
        images={},
        interval=1000,
        root_path="logs",
        summary_writer=None
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
                self.tb_logger.add_images(key, (value+1)/2, step, dataformats='NCHW')


class LODHook(Hook):
    def __init__(
        self,
        placeholder,
        schedule={
            4:[     0,  15000],
            3:[ 30000,  45000],
            2:[ 60000,  75000],
            1:[ 90000, 105000],
            0:[120000,1000000],
        },
        **kwargs
    ):
        self.schedule = schedule
        tmp = [[k, x] for (k, l) in schedule.items() for x in l]
        self.reduced_schedule = np.array(tmp).T

        self.step = 0

        self.pl = placeholder
        self.logger = get_logger(self)
        self.logger.info(self.reduced_schedule)


    def get_lod_from_step(self, step):
        idx = np.digitize(np.array([step]), self.reduced_schedule[1])-1
        lod_lo, lod_hi = self.reduced_schedule[0, idx], self.reduced_schedule[0, idx+1]
        start, end = self.reduced_schedule[1, idx], self.reduced_schedule[1, idx+1]
        curr_lod = linear_var(step, start, end, lod_lo, lod_hi, min(lod_lo, lod_hi), max(lod_lo, lod_hi))
        return curr_lod
            

    def before_step(self, batch_index, fetches, feeds, batch):
        batch['lod'] = self.get_lod_from_step(self.step)
        feeds[self.pl] = self.get_lod_from_step(self.step)


    def after_step(self, batch_index, last_results):
        self.step = last_results["global_step"]


class scoreLODHook(Hook):
    def __init__(
        self,
        placeholder,
        scalars,
        interval=1000,
        schedule={
            0:.05,
            1:.1,
            2:.25,
            3:.5,
            4:1,
        },
        root_path="logs",
        summary_writer=None,
    ):
        self.schedule = schedule
        tmp = [l for (k, l) in schedule.items()]
        self.reduced_schedule = np.array(tmp+[100])

        self.scalars = scalars
        self.keys = list(scalars.keys())
        self.interval = interval

        self.root = root_path
        self.logger = get_logger(self)
        if summary_writer is None:
            self.tb_logger = SummaryWriter(root_path)
        else:
            self.tb_logger = summary_writer

        self.results_log = {key:[10] for key in self.keys}
        self.prefix = 'lod/'

        layout = {'lod':{'scores':
            ['Multiline', 
            [
                self.prefix+'mean',
                self.prefix+'std',
            ]]
        }}

        self.tb_logger.add_custom_scalars(layout)
        self.step = 0

        self.pl = placeholder

        self.curr_lod = 4
        self.old_lod = 4


    def get_lod_from_scores(self):
        self.m = np.mean([self.results_log[key] for key in self.keys])
        self.s = np.std([self.results_log[key] for key in self.keys])

        idx = np.digitize(self.s, self.reduced_schedule) - 1
        idx = max(0, idx)
        idx = min(4, idx)

        self.curr_lod = idx


    def before_step(self, batch_index, fetches, feeds, batch):
        #fetches["scoreLOD"] = self.scalars
        fetches["lod_scalars"] = self.scalars

        self.get_lod_from_scores()

        if int(self.old_lod) != self.old_lod:
            end = self.start + 10000 * (4 - floor(self.old_lod))
            self.old_lod = linear_var(self.step, self.start, end, ceil(self.old_lod), floor(self.old_lod), floor(self.old_lod), ceil(self.old_lod))

        elif self.step % self.interval == 0:
            if self.curr_lod < self.old_lod:
                self.old_lod -= .0001
                self.start = self.step
            elif self.curr_lod > self.old_lod:
                self.old_lod = self.old_lod

        feeds[self.pl] = self.old_lod


    def after_step(self, batch_index, last_results):
        self.step = last_results["global_step"]
        results = last_results["lod_scalars"]
        for (key, value) in results.items():
            self.results_log[key].append(value)
            if len(self.results_log[key]) > self.interval:
                self.results_log[key].pop(0)

        if batch_index % self.interval == 0:
            self.tb_logger.add_scalar(self.prefix+'mean', self.m, self.step)
            self.tb_logger.add_scalar(self.prefix+'std', self.s, self.step)
            self.tb_logger.add_scalar(self.prefix+'curr_lod', self.curr_lod, self.step)
            self.logger.info(f"curr_lod: {self.curr_lod}")
