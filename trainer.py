from edflow.iterators.tf_trainer import TFListTrainer
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.hooks.hook import Hook
from edflow.hooks.util_hooks import IntervalHook
from edflow.hooks.logging_hooks.tf_logging_hook import LoggingHook
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import CheckpointHook
from edflow.util import pprint, retrieve
from edflow.custom_logging import get_logger
from edflow.project_manager import ProjectManager
from edflow.tf_util import make_linear_var


from .loss import G_logistic_nonsaturating, D_logistic
from .misc import process_reals
from .hooks import *
import stylegan.ops as op

import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter

class ListTrainer(TFListTrainer):
    def setup(self):
        self.accuracy = 0.8
        self.alpha = 0.05
        self.win_rate = self.config.get("win_rate", .8)

        self.curr_phase = 'discr'

        self.train_placeholders = dict()
        self.log_ops = dict()
        self.img_ops = dict()
        self.hist_ops = dict()
        self.s_ops = dict()
        self.update_ops = list()
        self.create_train_op()

        tb_writer = SummaryWriter(ProjectManager.train)

        ckpt_hook = CheckpointHook(
            root_path=ProjectManager.checkpoints,
            variables=self.get_checkpoint_variables(),
            modelname="model",
            step=self.get_global_step,
            interval=self.config.get("ckpt_freq", None),
            max_to_keep=self.config.get("ckpt_keep", None),
            )
        self.hooks.append(ckpt_hook)

        lodhook = scoreLODHook(self.lod_in,
                scalars = {k:v for k, v in self.s_ops.items() if k in ['scores/real', 'scores/fake']},
            #schedule={
            #    4:[     0,  15000],
            #    3:[ 30000,  45000],
            #    2:[ 60000,  75000],
            #    1:[ 90000, 105000],
            #    0:[120000,1000000],
            #},
        )
        self.hooks.append(lodhook)

        loghook = MarginPlottingHook(
            scalars = self.s_ops,
            interval = self.config.get("log_freq", 1000),
            root_path=ProjectManager.train,
            summary_writer = tb_writer,
            )
        self.hooks.append(loghook)

        imghook = ImageLoggingHook(
            images = self.img_ops,
            interval = self.config.get("log_freq", 1000),
            root_path=ProjectManager.train,
            summary_writer = tb_writer,
            )
        self.hooks.append(imghook)


    def define_connections(self):
        batch_size = self.config["batch_size"]
        mirror_augment = self.config.get('mirror_augment', True)
        drange_net = self.config.get('drange_net', [-1, 1])

        with tf.name_scope('placeholder'):
            dtype = self.config.get('dtype', tf.float32)
            latents_in = tf.placeholder(dtype=dtype,
                                        shape=[batch_size, None],
                                        name='latents_in')
            labels_in = tf.placeholder(dtype=dtype,
                                       shape=[batch_size, None],
                                       name='labels_in')
            images_in = tf.placeholder(dtype=dtype,
                                       shape=[batch_size, None, None, None],
                                       name='images_in')
            lod_in = tf.placeholder(dtype=dtype,
                                    shape=[],
                                    name='lod_in')


        global_step = self._global_step_variable * batch_size
        self.s_ops['lod'] = lod_in
        self.lod_in = lod_in

        images_out = self.model.generate(latents_in, labels_in, lod_in)

        fake_scores_out, _ = self.model.discriminate(images_out, labels_in, lod_in)
        #images_in = process_reals(images_in, lod_in, mirror_augment, [-1, 1], drange_net)

        real_scores_out, real_scaled = self.model.discriminate(images_in, labels_in, lod_in)

        self.model.outputs = {'images_out': images_out, 'scaled_images':real_scaled}

        self.model.scores = {
            'fake_scores_out': fake_scores_out,
            'real_scores_out': real_scores_out,
            }

        #self.model.inputs = {'latent':latents_in, 'feature_vec':labels_in, 'image':images_in}
        self.model.inputs = {
            'latent':latents_in,
            'painted':labels_in,
            'image':images_in,
            'lod':lod_in
            }

        self.model.variables = tf.global_variables()

    def make_loss_ops(self):
        self.define_connections()

        gen_loss = G_logistic_nonsaturating(self.model.scores['fake_scores_out'])
        discr_loss = D_logistic(
            self.model.scores['real_scores_out'],
            self.model.scores['fake_scores_out'])

        self.img_ops['fake'] = self.model.outputs['images_out']
        self.img_ops['real'] = self.model.outputs['scaled_images']

        self.s_ops['losses/gen'] = tf.reduce_mean(gen_loss)
        self.s_ops['losses/discr'] = tf.reduce_mean(discr_loss)
        self.s_ops['scores/fake'] = tf.reduce_mean(self.model.scores['fake_scores_out'])
        self.s_ops['scores/real'] = tf.reduce_mean(self.model.scores['real_scores_out'])

        losses = []
        losses.append({"generator": gen_loss})
        losses.append({"discriminator": discr_loss})
        
        return losses

    def run(self, fetches, feed_dict):
        #self.logger.info(feed_dict)
        #self.logger.info(fetches)
        result = super().run(fetches, feed_dict)
        #pprint(result)
        return result

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

