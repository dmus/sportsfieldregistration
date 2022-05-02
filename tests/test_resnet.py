import tensorflow as tf
from classification_models.tfkeras import Classifiers

from sportsfieldregistration.layers import SpectralNormalization

ResNet18, _ = Classifiers.get('resnet18')
base_model = ResNet18(input_shape=(720, 1280, 6), weights=None, include_top=False)

# Add spectral normalization
base_model_new = tf.keras.models.Sequential()
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        spec_norm_layer = SpectralNormalization(layer)
        base_model_new.add(spec_norm_layer)
    else:
        base_model_new.add(layer)

input = tf.keras.Input(shape=(720, 1280, 3), name='image')
x = base_model_new(input)

model = tf.keras.models.Model(inputs=[input], outputs=x, name='test')