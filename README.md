# Squeeze and Excitation Networks in Keras
Implementation of [Squeeze and Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) in Keras 2.0.3+.

<img src="https://github.com/titu1994/keras-squeeze-excite-network/blob/master/images/squeeze-excite-block.JPG?raw=true" height=50% width=100% alt="squeeze-excite-block">

## Models
Current models supported :

- SE-ResNet. Custom ResNets can be built using the `SEResNet` model builder, whereas prebuilt Resnet models such as `SEResNet50`, `SEResNet101` and `SEResNet154` can also be built directly.
- SE-InceptionV3
- SE-Inception-ResNet-v2
- SE-ResNeXt

Additional models (not from the paper, not verified if they improve performance)
- SE-MobileNets
- SE-DenseNet - Custom SE-DenseNets can be built using `SEDenseNet` model builder, whereas prebuilt SEDenseNet models such as `SEDenseNetImageNet121`, `SEDenseNetImageNet169`, `SEDenseNetImageNet161`, `SEDenseNetImageNet201` and `SEDenseNetImageNet264` can be build DenseNet in ImageNet configuration. To use SEDenseNet in CIFAR mode, use the `SEDenseNet` model builder.

## Squeeze and Excitation block
The block is simple to implement in Keras. It composes of a GlobalAveragePooling2D, 2 Dense blocks and an elementwise multiplication.
Shape inference can be done automatically in Keras. It can be imported from `se.py`.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
```


## Addition of Squeeze and Excitation blocks to Inception and ResNet blocks
<img src="https://github.com/titu1994/keras-squeeze-excite-network/blob/master/images/se-architectures.jpg?raw=true" height=50% width=50% alt="se-architectures"> <img src="https://github.com/titu1994/keras-squeeze-excite-network/blob/master/images/SE-ResNet-architecture.jpg?raw=true" height=50% width=49% alt="SE-ResNet-architecture">


