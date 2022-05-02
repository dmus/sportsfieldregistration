import datetime
import os

import tensorflow as tf

from sportsfieldregistration.metrics import IOUPart
# from spectral_normalization import SpectralNormalization
from sportsfieldregistration.models import registration_error_model
from sportsfieldregistration.utils import get_perspective_transform, warp_image, get_dataset, create_errors


def loss_fun(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    input_shape = (360, 640, 3)
    model = registration_error_model(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=loss_fun, optimizer=optimizer, run_eagerly=True)
    #
    metric = IOUPart()
    dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=4, shuffle=True)
    dataset_train = dataset_train.map(lambda images, labels: create_errors(images, labels, metric)).take(8)
    dataset_train = dataset_train.prefetch(1)

    dataset_val = get_dataset('/home/derk/sports-field-registration/raw/test', batch_size=4, shuffle=True)
    dataset_val = dataset_val.map(lambda images, labels: create_errors(images, labels, metric)).take(8)
    dataset_val = dataset_val.prefetch(1)

    experiment_path = os.path.join('/home/derk/sports-field-registration/results_registration_error',
                                   '{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    filepath = os.path.join(experiment_path, 'checkpoint-{epoch:02d}.h5')  # {val_loss:.2f}.h5')

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(experiment_path)),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, monitor='val_loss', save_best_only=True)
        # tf.keras.callbacks.LearningRateScheduler()
    ]

    model.fit(dataset_train, epochs=200, validation_data=dataset_train, callbacks=callbacks)

    print('Evaluation')
    scores_train = model.evaluate(dataset_train)
    scores_val = model.evaluate(dataset_val)
    print(scores_train)
    print(scores_val)


if __name__ == '__main__':
    main()
