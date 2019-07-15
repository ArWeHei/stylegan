from edflow.iterators.tf_trainer import TFListTrainer
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.hooks.hook import Hook
from edflow.hooks.util_hooks import IntervalHook
from edflow.hooks.logging_hooks.tf_logging_hook import LoggingHook
from edflow.hooks.logging_hooks.tf_logging_hook import ImageOverviewHook
from edflow.util import pprint, retrieve
from edflow.custom_logging import get_logger
from edflow.project_manager import ProjectManager

from .loss import G_logistic_nonsaturating, D_logistic

import tensorflow as tf
import numpy as np

class ListTrainer(TFListTrainer):
    def setup(self):
        self.accuracy = 0.8
        self.alpha = 0.05
        self.win_rate = self.config.get("win_rate", .8)

        self.define_connections()

        self.curr_phase = 'discr'

        super().setup()


    def define_connections(self):
        batch_size = self.config["batch_size"]

        with tf.name_scope('placeholder'):
            dtype = self.config.get('dtype', tf.float32)
            latents_in = tf.placeholder(dtype=dtype,
                                       shape=[None, None],
                                       name='latents_in')
            labels_in = tf.placeholder(dtype=dtype,
                                       shape=[None, None],
                                       name='labels_in')
            images_in = tf.placeholder(dtype=dtype,
                                       shape=[None, None, None, None],
                                       name='images_in')


        
        images_out = self.model.generate(latents_in, labels_in)

        fake_scores_out = self.model.discriminate(images_out, labels_in)
        real_scores_out = self.model.discriminate(images_in, labels_in)


        self.model.outputs = {'images_out': images_out}

        self.model.scores = {
            'fake_scores_out': fake_scores_out,
            'real_scores_out': real_scores_out,
            }

        self.model.inputs = {'latent':latents_in, 'features_vec':labels_in, 'image':images_in}

        self.model.variables = tf.global_variables()

    def make_loss_ops(self):

        gen_loss = G_logistic_nonsaturating(self.model.scores['fake_scores_out'])
        discr_loss = D_logistic(
            self.model.scores['real_scores_out'],
            self.model.scores['fake_scores_out'])


        losses = []
        losses.append({"generator": gen_loss})
        losses.append({"discriminator": discr_loss})
        
        return losses

    def run(self, fetches, feed_dict):
        self.logger.info(fetches)
        self.logger.info(feed_dict)
        super().run(fetches, feed_dict)
    #def run(self, fetches, feed_dict):
    #    if self.curr_phase == 'discr':
    #        adj = 0.05
    #    else:
    #        adj = -0.05
    #    adj_win_rate = self.win_rate + adj

    #    #decide in run when to switch optimizers
    #    if self.DSloss > self.DSloss_clip:
    #        train_idx = 0
    #        fetches["gen_steps"] = self.gen_steps
    #        self.curr_phase = 'gen'
    #        self.current_min_loss = self.min_DSloss
    #    elif self.accuracy >= adj_win_rate:
    #        train_idx = 0
    #        fetches["gen_steps"] = self.gen_steps
    #        self.curr_phase = 'gen'
    #        self.DSloss_clip = self.max_DSloss
    #    else:
    #        train_idx = 1
    #        fetches["discr_steps"] = self.discr_steps
    #        self.curr_phase = 'discr'
    #        self.DSloss_clip = self.max_DSloss

    #    fetches["step_ops"] = self.all_train_ops[train_idx]
    #    fetches["accuracy"] = self.acc
    #    fetches["DSloss"] = self.DSloss_op

    #    tmp = super(TFListTrainer, self).run(fetches, feed_dict)

    #    self.accuracy = self.accuracy * (1. - self.alpha) + self.alpha * tmp["accuracy"]
    #    self.DSloss = tmp["DSloss"]

    #    #for logging
    #    tmp['acc'] = {'smoothed_acc': self.accuracy}

    #    return tmp

