"""Network architectures used in the StyleGAN paper."""

import numpy as np
import tensorflow as tf

import stylegan.ops as ops
import stylegan.util as util

from edflow.util import pprint

def G_style(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    lod_in,
    truncation_psi          = 0.7,          # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = 8,            # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,         # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,         # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,        # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,          # Probability of mixing styles during training. None = disable.
    components              = None,         # Container for sub-networks. Retained between calls.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    resolution              = 128,          # Output resolution
    #num_layers              = 3,            # Number of output color channels.
    **kwargs):                              # Arguments for sub-networks (G_mapping and G_synthesis).
    # Setup components.
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 2
    if 'synthesis' not in components.keys():
        obj0 = globals()['G_synthesis']
        components['synthesis'] = lambda dlatents, lod_in: obj0(dlatents, lod_in, resolution=resolution, dlatent_size=dlatent_size, **kwargs)
    if 'mapping' not in components:
        obj1 = globals()['G_mapping']
        components['mapping'] = lambda latents_in, labels_in: obj1(latents_in, labels_in, dlatent_size=dlatent_size, dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    #lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    with tf.variable_scope('G_mapping'):
        dlatents = components['mapping'](latents_in, labels_in)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, util.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if 0:#if style_mixing_prob is not None:
        with tf.name_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            with tf.variable_scope('G_mapping', reuse=True):
                dlatents2 = components['mapping'](latents2, labels_in)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = util.lerp(dlatent_avg, dlatents, coefs)

    # Evaluate synthesis network.
    #with tf.control_dependencies([tf.assign(components['synthesis'].find_var('lod'), lod_in)]):
    with tf.variable_scope('G_synthesis'):
        images_out = components['synthesis'](dlatents, lod_in)
    return tf.identity(images_out, name='images_out')


def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu'.
    use_wscale              = True,         # Enable equalized learning rate?
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **kwargs):                             # Ignore unrecognized keyword args.

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        x = ops.pixel_norm(x)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = ops.dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = ops.apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


def G_synthesis(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    lod_in,
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_styles          = True,         # Enable style inputs?
    const_input_layer   = True,         # First layer is a learned constant?
    use_noise           = True,         # Enable noise inputs?
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
    use_instance_norm   = True,         # Enable instance normalization?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x):
        return ops.blur2d(x, blur_filter) if blur_filter else x

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]
    num_layers = resolution_log2 * 2 - 2
    num_styles = num_layers if use_styles else 1
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_styles, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    #lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
    #lod_in = lod

    # Noise inputs.
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, use_noise, 2**res, 2**res]
            noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Things to do at the end of each layer.
    def layer_epilogue(x, layer_idx):
        if use_noise:
            x = ops.apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
        x = ops.apply_bias(x)
        x = act(x)
        if use_pixel_norm:
            x = ops.pixel_norm(x)
        if use_instance_norm:
            x = ops.instance_norm(x)
        if use_styles:
            x = ops.style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
        return x

    # Early layers.
    with tf.variable_scope('4x4'):
        if const_input_layer:
            with tf.variable_scope('Const'):
                x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())
                x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1]), 0)
        else:
            with tf.variable_scope('Dense'):
                x = ops.dense(dlatents_in[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
                x = layer_epilogue(tf.reshape(x, [-1, nf(1), 4, 4]), 0)
        with tf.variable_scope('Conv'):
            x = layer_epilogue(ops.conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=use_wscale), 1)

    # Building blocks for remaining layers.
    def block(res, x): # res = 3..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                x = layer_epilogue(blur(ops.upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)), res*2-4)
            with tf.variable_scope('Conv1'):
                x = layer_epilogue(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
            return x
    def torgb(res, x): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod, reuse=tf.AUTO_REUSE):
            return tf.nn.tanh(ops.apply_bias(ops.conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale)))

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = torgb(resolution_log2, x)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        images_out = torgb(2, x)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(res, x)
            img = torgb(res, x)
            images_out = ops.upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = util.lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: ops.upscale2d(torgb(res, y), 2**lod)
            img = cset(img, (lod_in > lod), lambda: ops.upscale2d(util.lerp(torgb(res, y), ops.upscale2d(torgb(res - 1, x)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


def D_basic(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    lod_in,
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    **kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x):
        return ops.blur2d(x, blur_filter) if blur_filter else x
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    #lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    scores_out = None

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res), reuse=tf.AUTO_REUSE):
            return act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Conv1_down'):
                    x = act(ops.apply_bias(ops.conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
            else: # 4x4
                if mbstd_group_size > 1:
                    x = ops.minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(ops.apply_bias(ops.dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = ops.apply_bias(ops.dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        scores_out = block(x, 2)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = ops.downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = util.lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, 2)

    # Recursive structure: complex but efficient.
    scaled_img = images_in
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            x = lambda: fromrgb(ops.downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: util.lerp(x, fromrgb(ops.downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        def d_scale(lod):
            x = lambda: ops.upscale2d(ops.downscale2d(images_in, 2**lod), 2**lod)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: d_scale(lod - 1))
            y = cset(x, (lod_in > lod), lambda: util.lerp(x(), ops.upscale2d(ops.downscale2d(images_in, 2**(lod+1)), 2**(lod+1)), lod_in - lod))
            return y()

        scores_out = grow(2, resolution_log2 - 2)
        scaled_img = d_scale(resolution_log2 - 2)

    # Label conditioning from "Which Training Methods for GANs do actually Converge?"
    if label_size:
        with tf.variable_scope('LabelSwitch'):
            #scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)
            print('====================================================')
            print(scores_out)
            print(labels_in)
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out, scaled_img

def D_style(
    images_in,
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    lod_in,
    truncation_psi          = 0.7,          # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = 8,            # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,         # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,         # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,        # Decay for tracking the moving average of W during training. None = disable.
    components              = None,         # Container for sub-networks. Retained between calls.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    resolution              = 128,          # Output resolution
    #num_layers              = 3,            # Number of output color channels.
    **kwargs):                              # Arguments for sub-networks (G_mapping and G_synthesis).
    # Setup components.
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 2
    if 'synthesis' not in components.keys():
        obj0 = globals()['D_synthesis']
        components['synthesis'] = lambda images_in, dlatents, lod_in: obj0(images_in, dlatents, lod_in, resolution=resolution, dlatent_size=dlatent_size, **kwargs)
    if 'mapping' not in components:
        obj1 = globals()['D_mapping']
        components['mapping'] = lambda labels_in: obj1(labels_in, dlatent_size=dlatent_size, dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    #lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    with tf.variable_scope('D_mapping'):
        dlatents = components['mapping'](labels_in)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, util.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Apply truncation trick.
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = util.lerp(dlatent_avg, dlatents, coefs)

    # Evaluate synthesis network.
    #with tf.control_dependencies([tf.assign(components['synthesis'].find_var('lod'), lod_in)]):
    with tf.variable_scope('D_synthesis'):
        scores_out, scaled_imgs = components['synthesis'](images_in, dlatents, lod_in)
    return tf.identity(scores_out, name='scores_out'), scaled_imgs


def D_mapping(
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu'.
    use_wscale              = True,         # Enable equalized learning rate?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **kwargs):                             # Ignore unrecognized keyword args.

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # Inputs.
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)
    x = tf.get_variable('mapping_const', shape=[label_size+1], initializer=tf.initializers.random_normal())

    # Embed labels and concatenate them with latents.
    if label_size:
        x = labels_in
    else:
        x = tf.get_variable('mapping_const', shape=[1], initializer=tf.initializers.random_normal())

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = ops.dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = ops.apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

def D_synthesis(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    dlatents_in,                          # Second input: Labels [minibatch, label_size].
    lod_in,
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    use_styles          = True,         # Enable style inputs?
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
    use_instance_norm   = True,         # Enable instance normalization?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    **kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x):
        return ops.blur2d(x, blur_filter) if blur_filter else x
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]
    num_layers = resolution_log2 * 2 - 2
    num_styles = num_layers if use_styles else 1

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    dlatents_in.set_shape([None, num_styles, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    scores_out = None


    # Things to do at the end of each layer.
    def layer_epilogue(x, layer_idx):
        x = ops.apply_bias(x)
        x = act(x)
        if use_pixel_norm:
            x = ops.pixel_norm(x)
        if use_instance_norm:
            x = ops.instance_norm(x)
        if use_styles:
            x = ops.style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
        return x

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res), reuse=tf.AUTO_REUSE):
            return act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = layer_epilogue(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-4)
                with tf.variable_scope('Conv1_down'):
                    x = layer_epilogue(ops.conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale), res*2-3)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = ops.minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    x = layer_epilogue(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
                with tf.variable_scope('Dense0'):
                    x = act(ops.apply_bias(ops.dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = ops.apply_bias(ops.dense(x, fmaps=1, gain=1, use_wscale=use_wscale))
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        scores_out = block(x, 2)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = ops.downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = util.lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, 2)

    # Recursive structure: complex but efficient.
    scaled_img = images_in
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            #this line....
            x = lambda: fromrgb(ops.downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            #modify s.t. there is no fading in the discriminator
            #....and this line do not correlate enough. The network is fed with unknown weigths > bad
            if res > 2: y = cset(y, (lod_in > lod), lambda: util.lerp(x, fromrgb(ops.downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        def d_scale(lod):
            x = lambda: ops.upscale2d(ops.downscale2d(images_in, 2**lod), 2**lod)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: d_scale(lod - 1))
            y = cset(x, (lod_in > lod), lambda: util.lerp(x(), ops.upscale2d(ops.downscale2d(images_in, 2**(lod+1)), 2**(lod+1)), lod_in - lod))
            return y()

        scores_out = grow(2, resolution_log2 - 2)
        scaled_img = d_scale(resolution_log2 - 2)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out, scaled_img


def Q_basic(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    lod_in,
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 10,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    use_wscale          = True,         # Enable equalized learning rate?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'recursive',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x):
        return ops.blur2d(x, blur_filter) if blur_filter else x
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res), reuse=tf.AUTO_REUSE):
            return act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv_down'):
                    x = act(ops.apply_bias(ops.conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
            else: # 4x4
                with tf.variable_scope('Conv'):
                    x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(ops.apply_bias(ops.dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = ops.apply_bias(ops.dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        scores_out = block(x, 2)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = ops.downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = util.lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, 2)

    # Recursive structure: complex but efficient.
    scaled_img = images_in
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            x = lambda: fromrgb(ops.downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: util.lerp(x, fromrgb(ops.downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()

        scores_out = grow(2, resolution_log2 - 2)
        print(scores_out)

    # Label conditioning from "Which Training Methods for GANs do actually Converge?"
    #if label_size:
    #    with tf.variable_scope('LabelSwitch'):
    #        #scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)
    #        scores_out = tf.reduce_sum(scores_out, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


