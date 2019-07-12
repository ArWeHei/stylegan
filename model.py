"""Network architectures used in the StyleGAN paper."""

import numpy as np
import tensorflow as tf

from edflow.util import pprint

import stylegan.ops as ops
import stylegan.util as util
import stylegan.networks as networks

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(
        *args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items())
    )
    return tf.make_template(name, run, unique_name_=name)

#----------------------------------------------------------------------------
# Style-based generator used in the StyleGAN paper.
# Composed of two sub-networks (G_mapping and G_synthesis) that are defined below.

class generator(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = dict()

        self.kwargs['truncation_psi']      = config.get('truncation_psi',         0.7)   # Style strength multiplier for the truncation trick. None = disable.
        self.kwargs['truncation_cutoff']   = config.get('truncation_cutoff',      8)     # Number of layers for which to apply the truncation trick. None = disable.
        self.kwargs['dlatent_avg_beta']    = config.get('dlatent_avg_beta',       0.995) # Decay for tracking the moving average of W during training. None = disable.
        self.kwargs['style_mixing_prob']   = config.get('style_mixing_prob',      0.9)   # Probability of mixing styles during training. None = disable.
        self.kwargs['dlatent_size']        = config.get('dlatent_size',           512)          # Disentangled latent (W) dimensionality.
        self.kwargs['num_layers']          = config.get('num_channels',           3)            # Number of output color channels.

        self.kwargs['latent_size']             = config.get('latent_size',          512)          # Latent vector (Z) dimensionality.
        self.kwargs['label_size']              = config.get('label_size',           0)            # Label dimensionality, 0 if no labels.
        #self.kwargs['dlatent_broadcast']       = config.get('dlatent_broadcast',    None)         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
        self.kwargs['mapping_layers']          = config.get('mapping_layers',       8)            # Number of mapping layers.
        self.kwargs['mapping_fmaps']           = config.get('mapping_fmaps',        512)          # Number of activations in the mapping layers.
        self.kwargs['mapping_lrmul']           = config.get('mapping_lrmul',        0.01)         # Learning rate multiplier for the mapping layers.
        self.kwargs['mapping_nonlinearity']    = config.get('mapping_nonlinearity', 'lrelu')      # Activation function: 'relu', 'lrelu'.
        self.kwargs['normalize_latents']       = config.get('normalize_latents',    True)         # Normalize latent vectors (Z) before feeding them to the mapping layers?

        self.kwargs['resolution']          = config.get('resolution',         1024)         # Output resolution.
        self.kwargs['fmap_base']           = config.get('fmap_base',          8192)         # Overall multiplier for the number of feature maps.
        self.kwargs['fmap_decay']          = config.get('fmap_decay',         1.0)          # log2 feature map reduction when doubling the resolution.
        self.kwargs['fmap_max']            = config.get('fmap_max',           512)          # Maximum number of feature maps in any layer.
        self.kwargs['use_styles']          = config.get('use_styles',         True)         # Enable style inputs?
        self.kwargs['const_input_layer']   = config.get('const_input_layer',  True)         # First layer is a learned constant?
        self.kwargs['use_noise']           = config.get('use_noise',          True)         # Enable noise inputs?
        self.kwargs['randomize_noise']     = config.get('randomize_noise',    True)         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        self.kwargs['nonlinearity']        = config.get('nonlinearity',       'lrelu')      # Activation function: 'relu', 'lrelu'
        self.kwargs['use_pixel_norm']      = config.get('use_pixel_norm',     False)        # Enable pixelwise feature vector normalization?
        self.kwargs['use_instance_norm']   = config.get('use_instance_norm',  True)         # Enable instance normalization?
        self.kwargs['fused_scale']         = config.get('fused_scale',        'auto')       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        self.kwargs['blur_filter']         = config.get('blur_filter',        [1,2,1])      # Low-pass filter to apply when resampling activations. None = no filtering.
        self.kwargs['structure']           = config.get('structure',          'recursive')       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.

        self.kwargs['use_wscale']          = config.get('use_wscale',         True)         # Enable equalized learning rate?
        self.kwargs['dtype']               = config.get('dtype',              'float32')    # Data type to use for activations and outputs.

        self.name = "generator"

        self.define_graph()

    @property
    def inputs(self):
        return {'latents_in':self.latents_in, 'labels_in': self.labels_in}

    @property
    def outputs(self):
        return {'images_out': self.images_out}

    def define_graph(self):
        #with tf.name_scope('placeholder'):
        #    self.latents_in = tf.placeholder(dtype=self.kwargs['dtype'],
        #                               shape=[None, None],
        #                               name='latents_in')
        #    self.labels_in = tf.placeholder(dtype=self.kwargs['dtype'],
        #                               shape=[None, None],
        #                               name='labels_in')
    
        #components = {'synthesis':networks.G_synthesis, 'mapping':networks.G_mapping}
        components = dict()
        self.network = make_model(self.name, networks.G_style, components=components, **self.kwargs)
        #self.images_out = self.network(self.latents_in, self.labels_in)


    def __call__(self, latents_in, labels_in):
        return self.network(latents_in, labels_in)

    @property
    def variables(self):
        return  [v for v in tf.global_variables() if v.name.startswith(self.name)]

#----------------------------------------------------------------------------
# Discriminator used in the StyleGAN paper.
class discriminator(object):
    def __init__(self, config, **kwargs):
        self.config = config

        self.kwargs = dict()
        self.kwargs['num_channels']        = config.get('num_channels',       3)            # Number of input color channels. Overridden based on dataset.
        self.kwargs['resolution']          = config.get('resolution',         32)           # Input resolution. Overridden based on dataset.
        self.kwargs['label_size']          = config.get('label_size',         0)            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        self.kwargs['fmap_base']           = config.get('fmap_base',          8192)         # Overall multiplier for the number of feature maps.
        self.kwargs['fmap_decay']          = config.get('fmap_decay',         1.0)          # log2 feature map reduction when doubling the resolution.
        self.kwargs['fmap_max']            = config.get('fmap_max',           512)          # Maximum number of feature maps in any layer.
        self.kwargs['nonlinearity']        = config.get('nonlinearity',       'lrelu')      # Activation function: 'relu', 'lrelu',
        self.kwargs['use_wscale']          = config.get('use_wscale',         True)         # Enable equalized learning rate?
        self.kwargs['mbstd_group_size']    = config.get('mbstd_group_size',   4)            # Group size for the minibatch standard deviation layer, 0 = disable.
        self.kwargs['mbstd_num_features']  = config.get('mbstd_num_features', 1)            # Number of features for the minibatch standard deviation layer.
        self.kwargs['dtype']               = config.get('dtype',              'float32')    # Data type to use for activations and outputs.
        self.kwargs['fused_scale']         = config.get('fused_scale',        'auto')       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        self.kwargs['blur_filter']         = config.get('blur_filter',        [1,2,1])      # Low-pass filter to apply when resampling activations. None = no filtering.
        self.kwargs['structure']           = config.get('structure',          'recursive')       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.

        self.name = "discriminator"

        self.define_graph()

    @property
    def inputs(self):
        return {'images_in':self.images_in, 'labels_in': self.labels_in}

    @property
    def outputs(self):
        return {'scores_out': self.scores_out}

    def define_graph(self):
        #with tf.name_scope('placeholder'):
        #    self.images_in = tf.placeholder(dtype=self.kwargs['dtype'],
        #                               shape=[None, None, None, None],
        #                               name='images_in')
        #    self.labels_in = tf.placeholder(dtype=self.kwargs['dtype'],
        #                               shape=[None, None],
        #                               name='labels_in')
        self.network = make_model(self.name, networks.D_basic, **self.kwargs)
        #self.scores_out = self.network(self.images_in, self.labels_in)

    def __call__(self, images_in, labels_in):
        return self.network(images_in, labels_in)

    @property
    def variables(self):
        return  [v for v in tf.global_variables() if v.name.startswith(self.name)]


#----------------------------------------------------------------------------
class TrainModel(object):
    def __init__(self, config):
        self.config=config
        self.generator = generator(config)
        self.discriminator = discriminator(config)
        #TODO:self.perceptor = VGGModel(config)
        self.variables = {
            'generator':self.generator.variables,
            'discriminator':self.discriminator.variables,
                          }

    def generate(self, latents_in, labels_in):
        return self.generator(latents_in, labels_in)

    def discriminate(self, images_in, labels_in):
        return self.discriminator(images_in, labels_in)
