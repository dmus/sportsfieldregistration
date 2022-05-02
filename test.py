import tensorflow as tf

from sportsfieldregistration.metrics import IOUPart
from sportsfieldregistration.utils import create_errors, get_dataset, visualize, get_perspective_transform, warp_image
from train import loss_fun


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    metric = IOUPart()
    dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=1)
    dataset_train = dataset_train.map(lambda images, labels: create_errors(images, labels, metric))
    #dataset_train = dataset_train.take(8)
    dataset_val = get_dataset('/home/derk/sports-field-registration/raw/test', batch_size=1)
    dataset_val = dataset_val.map(lambda images, labels: create_errors(images, labels, metric))

    # for batch in dataset_train:
    #     input = batch[0]
    #     output = batch[1]
    #     import pdb;pdb.set_trace()

    custom_objects = {
        'loss_fun': loss_fun,
        'IOUPart': IOUPart
    }
    model = tf.keras.models.load_model('/home/derk/sports-field-registration/results_registration_error/20200208154422/checkpoint-109.h5', compile=False)  # 'results/20200206000558/checkpoint-170.h5'
    model.compile(optimizer='adam', loss=loss_fun, metrics=[], run_eagerly=True)

    model.summary()

    #scores_train = model.evaluate(dataset_train)
    #scores_val = model.evaluate(dataset_val)
    for input, error in dataset_val:
        homography = tf.convert_to_tensor(input[1])
        homography_variable = tf.Variable(homography)
        for iteration in range(400):
            homography = tf.convert_to_tensor(homography_variable.numpy())
            with tf.GradientTape() as tape:
                tape.watch(homography)
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                input = (input[0], homography, input[2], input[3])
                predicted_error = model(input, training=False)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                #loss_value = loss_fun(error, predicted_error)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(1 - predicted_error, homography)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer.apply_gradients(zip([grads], [homography_variable]))
            print('iteration {}'.format(iteration))
            #print('mean after apply_gradients: {}'.format(tf.reduce_mean(homography_variable)))

            visualize(input[0], homography)

if __name__ == '__main__':
    main()
