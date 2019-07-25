import tensorflow as tf
import numpy as np

import os
import time
import sys

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
            self.results_log[key] += [value]

        if batch_index % self.interval == 0:
            for key in self.keys:
                m = np.mean(self.results_log[key])
                s = np.std(self.results_log[key])
                self.m, self.s = m, s
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


class ImageEvalHook(Hook):
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
            4:[     0,  40000],
            3:[ 90000, 180000],
            2:[300000, 400000],
            1:[500000, 600000],
            0:[700000, 800000],
        },
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
        interval=100,
        schedule={
            4:[10., 1.0],
            3:[.75, .50],
            2:[.35, .25],
            1:[.15, .10],
            0:[.05, 0.0],
        },
    ):
        self.schedule = schedule
        tmp = [[k, x] for (k, l) in schedule.items() for x in l]
        self.reduced_schedule = np.array(tmp).T

        self.scalars = scalars
        self.keys = list(scalars.keys())
        self.interval = interval

        self.logger = get_logger(self)

        self.results_log = {key:[] for key in self.keys} #have a high initial value that drive the mean up
        self.scores = [10, 10]

        self.pl = placeholder

        self.logger.info(self.reduced_schedule)


    def get_lod_from_score(self, score):
        idx = np.digitize(np.array([score]), self.reduced_schedule[1])-1
        lod_lo, lod_hi = self.reduced_schedule[0, idx], self.reduced_schedule[0, idx+1]
        start, end = self.reduced_schedule[1, idx], self.reduced_schedule[1, idx+1]
        curr_lod = linear_var(score, start, end, lod_lo, lod_hi, min(lod_lo, lod_hi), max(lod_lo, lod_hi))
        return curr_lod
            

    def before_step(self, batch_index, fetches, feeds, batch):
        #batch['lod'] = self.get_lod_from_score(self.score)
        fetches["scoreLOD"] = self.scalars
        feeds[self.pl] = self.get_lod_from_score(np.mean(self.scores))


    def after_step(self, batch_index, last_results):
        step = last_results["global_step"]
        results = last_results["scoreLOD"]
        self.scores = []

        for (key, value) in results.items():
            self.results_log[key] += [value]
            if len(self.results_log[key]) >= self.interval:
                self.results_log[key] = self.results_log[keys][1:]
            self.scores.append(np.absolute(np.mean(self.results_log[key])))

        if step % self.interval == 0:
            self.logger.info(f'current log: {self.results_log}')
