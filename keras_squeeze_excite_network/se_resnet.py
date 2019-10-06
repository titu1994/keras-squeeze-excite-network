"""
Squeeze-and-Excitation ResNets

References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - []() # added when paper is published on Arxiv
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_squeeze_excite_network import TF

if TF:
    from tensorflow.keras import backend as K
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                                         GlobalAveragePooling2D, GlobalMaxPooling2D,
                                         Input, MaxPooling2D, add)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
    from tensorflow.python.keras.backend import is_keras_tensor
    from tensorflow.python.keras.utils import get_source_inputs
else:
    from keras import backend as K
    from keras.applications.resnet50 import preprocess_input
    from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                              GlobalAveragePooling2D, GlobalMaxPooling2D,
                              Input, MaxPooling2D, add)
    from keras.models import Model
    from keras.regularizers import l2
    from keras.applications.imagenet_utils import decode_predictions
    from keras.utils import get_source_inputs

    is_keras_tensor = K.is_keras_tensor

from keras_squeeze_excite_network.se import squeeze_excite_block
from keras_squeeze_excite_network.utils import _obtain_input_shape, _tensor_shape

__all__ = ['SEResNet', 'SEResNet50', 'SEResNet101', 'SEResNet154',
           'preprocess_input', 'decode_predictions']

WEIGHTS_PATH = ""
WEIGHTS_PATH_NO_TOP = ""


def SEResNet(input_shape=None,
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filter: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            width: width multiplier for the network (for Wide ResNets)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, bottleneck, weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

    # load weights

    return model


def SEResNet18(input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    depth=[2, 2, 2, 2],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet34(input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 4, 6, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet50(input_shape=None,
               width=1,
               bottleneck=True,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet101(input_shape=None,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 6, 23, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet154(input_shape=None,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 8, 36, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def _resnet_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block without bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block with bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, bottleneck, weight_decay, pooling):
    """Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x
