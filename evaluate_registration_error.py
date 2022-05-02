import tensorflow as tf
import matplotlib.pyplot as plt

from sportsfieldregistration.metrics import IOUPart
from sportsfieldregistration.utils import get_dataset, create_errors, get_perspective_transform, warp_image, visualize
from train_registration_error import loss_fun


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.random.set_seed(1234)
    metric = IOUPart()

    dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=1)
    dataset_train = dataset_train.map(lambda images, labels: create_errors(images, labels, metric)).take(8)
    dataset_val = get_dataset('/home/derk/sports-field-registration/raw/test', batch_size=1)
    dataset_val = dataset_val.map(lambda images, labels: create_errors(images, labels, metric)).take(8)

    model = tf.keras.models.load_model(
        '/home/derk/sports-field-registration/results_registration_error/20200301145604/checkpoint-74.h5',
        compile=False)  # 'results/20200206000558/checkpoint-170.h5'
    model.compile(optimizer='adam', loss=loss_fun, metrics=['mean_absolute_error'], run_eagerly=True)

    scores_train = model.evaluate(dataset_train)
    scores_val = model.evaluate(dataset_val)

    for input, target in dataset_train:
        prediction = model.predict(input)
        print('Target: {}, Predicted: {}'.format(target, prediction))

        # Input image
        plt.subplot(2, 1, 1)
        plt.imshow(input[0][0])

        # Input template warped
        homography = tf.transpose(tf.reshape(input[1][0], (-1, 2, 4)), [0, 2, 1])
        homography = get_perspective_transform(input[2], homography)

        # Homography matrix to warped template
        out_shape = (720, 1280, 3)
        warped_image = warp_image(input[3][0], homography, out_shape=out_shape)

        plt.subplot(2, 1, 2)
        plt.imshow(tf.squeeze(warped_image))

        plt.show()

        # image, warped template, iou, predicted iou
        # overfitting test
        #visualize(input[0], input[1])

if __name__ == '__main__':
    main()
