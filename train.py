import datetime
import os
import tensorflow as tf

from sportsfieldregistration.utils import get_dataset, random_flip, visualize
from sportsfieldregistration.models import initial_registration_model

def loss_fun(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    input_shape = (360, 640, 3)
    model = initial_registration_model(input_shape)

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=loss_fun, metrics=[], run_eagerly=False)

    dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', shuffle=True, batch_size=4).map(random_flip).prefetch(1)
    dataset_val = get_dataset('/home/derk/sports-field-registration/raw/test', batch_size=4)

    experiment_path = os.path.join('/home/derk/sports-field-registration/results', '{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    filepath = os.path.join(experiment_path, 'checkpoint-{epoch:02d}.h5') #{val_loss:.2f}.h5')

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(experiment_path)),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, monitor='val_loss', save_best_only=True)
        #tf.keras.callbacks.LearningRateScheduler()
    ]

    model.fit(dataset_train, validation_data=dataset_val, epochs=200, callbacks=callbacks)

    print('Evaluation')
    scores_train = model.evaluate(dataset_train)
    scores_val = model.evaluate(dataset_val)
    print(scores_train)
    print(scores_val)


if __name__ == '__main__':
    main()
