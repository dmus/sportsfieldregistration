import tensorflow as tf
from classification_models.tfkeras import Classifiers

from sportsfieldregistration.layers import SpectralNormalization
from sportsfieldregistration.utils import get_perspective_transform, warp_image


def initial_registration_model(input_shape):
    ResNet18, _ = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=input_shape, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(8)(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name='initial_registration')
    return model


def registration_error_model(input_shape):
    ResNet18, _ = Classifiers.get('resnet18')

    image_input = tf.keras.Input(shape=input_shape, name='image')
    h_input = tf.keras.Input(shape=(8,), name='homography')
    ref_corners_input = tf.keras.Input(shape=(4, 2), name='ref_corners')
    template_input = tf.keras.Input(shape=(680, 1050, 3), name='template')

    # 4 points to homography matrix
    x_homography = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.reshape(x, (-1, 2, 4)), [0, 2, 1]))(h_input)
    x_homography = tf.keras.layers.Lambda(lambda x: get_perspective_transform(x[0], x[1]))(
        [ref_corners_input, x_homography])

    # Homography matrix to warped template
    out_shape = input_shape
    x_warped_image = tf.keras.layers.Lambda(lambda x: warp_image(x[0], x[1], out_shape=out_shape))(
        [template_input, x_homography])

    x = tf.keras.layers.Concatenate(axis=-1)([image_input, x_warped_image])

    input_shape = (input_shape[0], input_shape[1], 6)
    base_model = ResNet18(input_shape=input_shape, weights=None, include_top=False)

    # Add spectral normalization
    # base_model_new = tf.keras.models.Sequential()
    # for layer in base_model.layers:
    #     if isinstance(layer, tf.keras.layers.Conv2D):
    #         base_model_new.add(SpectralNormalization(layer))
    #     else:
    #         base_model_new.add(layer)
    #
    # x = base_model_new(x)
    x = base_model(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[image_input, h_input, ref_corners_input, template_input], outputs=output,
                                  name='registration_error')

    return model
