import tensorflow as tf

from sportsfieldregistration.metrics import IOUPart, IOUWhole
from sportsfieldregistration.utils import get_dataset, visualize
from train import loss_fun


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=1)
    # dataset_train = dataset_train.take(8)
    dataset_val = get_dataset('/home/derk/sports-field-registration/raw/test', batch_size=1)

    custom_objects = {
        'loss_fun': loss_fun,
        'IOUPart': IOUPart
    }
    model = tf.keras.models.load_model('/home/derk/sports-field-registration/results/20200301131337/check,
                                       compile=False)
    model.compile(optimizer='adam', loss=loss_fun, metrics=[IOUPart()], run_eagerly=True)

    scores_train = model.evaluate(dataset_train)
    scores_val = model.evaluate(dataset_val)

    for image, target in dataset_val:
        prediction = model.predict(image)
        print(target)
        print(prediction)
        print('end')

        visualize(image, prediction)
        mse = (target - prediction) ** 2
        print('MSE: {}'.format(mse))


if __name__ == '__main__':
    main()
