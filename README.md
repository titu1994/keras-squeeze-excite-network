# Squeeze and Excitation Networks in Keras
Implementation of Squeeze and Excitation Networks (ResNet, WideResNet and Inception v3) in Keras 2.0+.

The paper has not come out yet, so verifying the code with paper is difficult, but this follows the diagrams posted by [hujie-frank : SENet](https://github.com/hujie-frank/SENet)

As and when the paper is published, the models will be updated to match the paper.

## Models
Currently supports all ResNet and WideResNet models and Inception V3. Custom ResNets can be built using the `SEResNet` model builder, whereas prebuilt Resnet models such as `SEResNet50`, `SEResNet101` and `SEResNet154` can also be built directly.

Support for Inception v4, Inception ResNet v2 models will come eventually.
