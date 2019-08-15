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

from .loss import *
from .misc import process_reals
from .hooks import *
import stylegan.ops as op

import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter

class ListTrainer(TFListTrainer):
    def setup(self):
        self.ema_alpha = self.config.get("ema_alpha", 0.005)
        self.curr_phase = 'discr'

        self.img_ops = dict()
        self.log_ops = dict()
        self.s_ops = dict()
        self.lod_scalar_ops = dict()
        self.train_placeholders = dict()
        self.update_ops = list()
        self.create_train_op()

        self.G_loss = 1
        self.D_loss = 1
        self.Q_loss = 1
        self.G_count = 0
        self.D_count = 0
        self.Q_count = 0

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


        LODschedule = self.config.get('LODschedule',
            {
                4:[     0,  10000],
                3:[ 20000,  40000],
                2:[ 80000, 160000],
                1:[320000, 640000],
                0:[100000,1000000],
            })
        lodhook = LODHook(self.lod_in,
            scalars = self.lod_scalar_ops,
            root_path=ProjectManager.train,
            summary_writer = tb_writer,
            schedule=LODschedule
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
        combine_features = self.config.get('combine_features', True)

        with tf.name_scope('placeholder'):
            dtype = self.config.get('dtype', tf.float32)
            latents_in = tf.placeholder(dtype=dtype,
                                        shape=[batch_size, None],
                                        name='latents_in')
            features_vec = tf.placeholder(dtype=dtype,
                                       shape=[batch_size, None],
                                       name='features_vec')
            painted = tf.placeholder(dtype=dtype,
                                       shape=[batch_size, None],
                                       name='painted')
            images_in = tf.placeholder(dtype=dtype,
                                       shape=[batch_size, None, None, None],
                                       name='images_in')
            lod_in = tf.placeholder(dtype=dtype,
                                    shape=[],
                                    name='lod_in')


        global_step = self._global_step_variable * batch_size
        self.s_ops['lod'] = lod_in
        self.lod_in = lod_in

        if combine_features:
            labels_in = tf.concat([features_vec, painted], axis=1)
        else:
            labels_in = features_vec
            #labels_in = painted

        images_out = self.model.generate(latents_in, labels_in, lod_in)

        eval_lat_in = np.repeat(np.random.standard_normal((batch_size//2, 512)), 2, axis=0)
        eval_lat_in = tf.constant(eval_lat_in)
        eval_painted = np.tile([[1,0,1], [0,1,-1]], (batch_size//2, 1))
        eval_painted = tf.constant(eval_painted, dtype=tf.float32)

        if combine_features:
            eval_lab_in = tf.concat([features_vec, eval_painted], axis=1)
        else:
            eval_lab_in = features_vec
            #eval_lab_in = painted

        eval_images_out = self.model.generate(eval_lat_in, eval_lab_in, lod_in)
        self.img_ops['eval'] = eval_images_out

        images_in = process_reals(images_in, lod_in, mirror_augment, [-1, 1], drange_net)
        real_scores_out, real_scaled, real_labels_out = self.model.discriminate(images_in, labels_in, lod_in)
        fake_scores_out, fake_scaled, fake_labels_out = self.model.discriminate(images_out, labels_in, lod_in)

        #fake_labels_out = self.model.classify(images_out, lod_in)
        #real_labels_out = self.model.classify(images_in, lod_in)

        self.model.outputs = {
            'images_out': images_out,
            'scaled_images':real_scaled,
            'fake_scaled_images':fake_scaled,
            }

        self.model.scores = {
            'fake_scores_out': fake_scores_out,
            'real_scores_out': real_scores_out,
            'fake_labels_out': fake_labels_out,
            'real_labels_out': real_labels_out,
            }

        #self.model.inputs = {'latent':latents_in, 'feature_vec':labels_in, 'image':images_in}
        self.model.inputs = {
            'latent':latents_in,
            'feature_vec':features_vec,
            'painted':painted,
            'labels_in':labels_in,
            'image':images_in,
            'lod':lod_in
            }

        self.model.variables = tf.global_variables()

    def make_loss_ops(self):
        self.define_connections()

        gen_loss = G_logistic_nonsaturating(self.model.scores['fake_scores_out'])
        print('===============================')
        pprint(self.optimizers)
        discr_loss = D_logistic_simplegp(
            self.model.scores['real_scores_out'],
            self.model.scores['fake_scores_out'],
            self.optimizers['discriminator']
            )
        class_loss = Q_sce(self.model.scores['fake_labels_out'],
                          (self.model.inputs['labels_in']+1) /2)
        gen_loss += class_loss
        class_loss += Q_sce(self.model.scores['real_labels_out'],
                         (self.model.inputs['labels_in']+1) /2)
        discr_loss += class_loss

        self.img_ops['fake'] = self.model.outputs['images_out']
        self.img_ops['real'] = self.model.outputs['scaled_images']
        self.img_ops['fake_scaled'] = self.model.outputs['fake_scaled_images']

        self.s_ops['losses/gen'] = tf.reduce_mean(gen_loss)
        self.s_ops['losses/discr'] = tf.reduce_mean(discr_loss)
        self.s_ops['losses/class'] = tf.reduce_mean(class_loss)
        self.s_ops['scores/fake'] = tf.reduce_mean(self.model.scores['fake_scores_out'])
        self.s_ops['scores/real'] = tf.reduce_mean(self.model.scores['real_scores_out'])
        self.lod_scalar_ops['fake'] = tf.reduce_mean(self.model.scores['fake_scores_out'])
        self.lod_scalar_ops['real'] = tf.reduce_mean(self.model.scores['real_scores_out'])

        losses = []
        losses.append({"generator": gen_loss})
        losses.append({"discriminator": discr_loss})
        #losses.append({"classifier": class_loss})

        g = tf.Variable(0, name="gen_step")
        d = tf.Variable(0, name="discr_step")
        q = tf.Variable(0, name="class_step")
        self.discr_steps = tf.assign_add(d, 1) 
        self.gen_steps = tf.assign_add(g, 1)
        self.class_steps = tf.assign_add(q, 1)
        self.s_ops["gen_steps"] = g
        self.s_ops["discr_steps"] = d
        #self.s_ops["class_steps"] = q

        return losses


    def create_train_op(self):
        """Default optimizer + optimize each submodule"""
        self.train_placeholders["global_step"] = tf.placeholder(
            tf.int32, name="global_step"
        )
        # workaround of https://github.com/tensorflow/tensorflow/issues/23316
        self._global_step_variable = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.global_step = tf.assign(
            self._global_step_variable, self.train_placeholders["global_step"]
        )

        # Optimizer
        self.initial_lr = self.config["lr"]
        self.ultimate_lr = self.config.get("lr_end", self.initial_lr)
        self.lr_decay_begin = self.config.get("lr_decay_begin", 0)
        self.lr_decay_end = self.config.get(
            "lr_decay_end", self.config.get("num_steps")
        )

        self.lr = lr = make_linear_var(
            self.global_step,
            self.lr_decay_begin,
            self.lr_decay_end,
            self.initial_lr,
            self.ultimate_lr,
            min(self.ultimate_lr, self.initial_lr),
            max(self.ultimate_lr, self.initial_lr),
        )
        optimizer_name = self.config.get("optimizer", "AdamOptimizer")
        Optimizer = getattr(tf.train, optimizer_name)
        opt_kwargs = {"learning_rate": lr}
        if optimizer_name == "AdamOptimizer":
            opt_kwargs["beta1"] = self.config.get("beta1", 0.5)
            opt_kwargs["beta2"] = self.config.get("beta2", 0.9)

        self.optimizers = dict()
        self.all_train_ops = list()

        for k in ['generator', 'discriminator']:
            current_opt_kwargs = dict(opt_kwargs)
            current_opt_kwargs["learning_rate"] = current_opt_kwargs[
                "learning_rate"
            ] * self.get_learning_rate_multiplier(i)
            self.optimizers[k] = Optimizer(**current_opt_kwargs)

        # Optimization ops
        list_of_losses = self.make_loss_ops()
        assert type(list_of_losses) == list
        for i, losses in enumerate(list_of_losses):
            opt_ops = dict()
            optimizers = dict()

            for k in losses:
                variables = self.get_trainable_variables(k)
                opt_ops[k] = self.optimizers[k].minimize(losses[k], var_list=variables)
                print(i, k, self.get_learning_rate_multiplier(i))
                print("============================")
                print(
                    "\n".join(
                        [
                            "{:>22} {:>22} {}".format(
                                str(v.shape.as_list()),
                                str(np.prod(v.shape.as_list())),
                                v.name,
                            )
                            for v in variables
                        ]
                    )
                )
                print(len(variables))
                print("============================")

            opt_op = tf.group(*opt_ops.values())
            with tf.control_dependencies([opt_op] + self.update_ops):
                train_op = tf.no_op()
            self.all_train_ops.append(train_op)

        self.train_op = None  # dummy, set actual train op in run
        self.run_once_op = self.make_run_once_op()
        # add log ops
        self.log_ops["global_step"] = self.global_step
        self.log_ops["lr"] = self.lr


    def make_run_once_op(self):
        print('===============================')
        print(self.optimizers)
        return tf.no_op()

    def run(self, fetches, feed_dict):
        print('===============================')
        pprint(self.optimizers)
        ##decide in run when to switch optimizers
        #if self.D_loss/self.G_loss > 2:
        #    self.curr_phase = 'discr'
        #elif self.G_loss/self.D_loss > 2:
        #    self.curr_phase = 'gen'
        ##if self.D_loss < self.G_loss:
        ##    self.curr_phase = 'gen'
        #elif self.curr_phase == 'discr':
        #    self.curr_phase = 'gen'
        #elif self.curr_phase == 'gen':
        #    self.curr_phase = 'discr'

        dec_values = [0,
                      self.G_loss,
                      self.D_loss,]
                     # self.Q_loss]

        dec_boundaries = np.cumsum(dec_values)
        dec_boundaries = dec_boundaries/np.sum(dec_values)

        r = np.random.uniform()

        idx = np.digitize(r, dec_boundaries) - 1

        if idx == 0:
            self.curr_phase = 'gen'
        elif idx == 1:
            self.curr_phase = 'discr'
        elif idx == 2:
            self.curr_phase = 'class'

        #if self.G_count >= 100:
        #    self.curr_phase = 'discr'
        #elif self.D_count >= 100:
        #    self.curr_phase = 'gen'

        if self.curr_phase == 'discr':
            train_idx = 1
            fetches["d_steps"] = self.discr_steps
            self.D_count += 1
            self.G_count = 0
            self.Q_count = 0
        elif self.curr_phase == 'gen':
            train_idx = 0
            fetches["g_steps"] = self.gen_steps
            self.G_count += 1
            self.D_count = 0
            self.Q_count = 0
        elif self.curr_phase == 'class':
            train_idx = 2
            fetches["q_steps"] = self.class_steps
            self.Q_count += 1
            self.D_count = 0
            self.G_count = 0

        fetches["step_ops"] = self.all_train_ops[train_idx]

        tmp = super(TFListTrainer, self).run(fetches, feed_dict)

        a = self.ema_alpha

        self.D_loss = a * tmp["custom_scalars"]["losses/discr"] + (1 - a)*self.D_loss
        self.G_loss = a * tmp["custom_scalars"]["losses/gen"] + (1 - a)*self.G_loss
        #self.Q_loss = a * tmp["custom_scalars"]["losses/class"] + (1 - a)*self.Q_loss

        return tmp

