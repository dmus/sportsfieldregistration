import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def loss_fun(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss

# Create a dataset
x0 = np.random.rand(1, 180, 320, 3).astype(np.float32)
x1 = np.random.rand(1, 180, 320, 3).astype(np.float32)
y = np.random.rand(1, 1).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices(((x0, x1), y)).batch(1)

# Create a model
input0 = tf.keras.layers.Input(shape=(180, 320, 3))
input1 = tf.keras.layers.Input(shape=(180, 320, 3))
x = tf.keras.layers.Concatenate()([input0, input1])
x = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(x)
x = tf.keras.applications.MobileNet(weights=None, include_top=False)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(inputs=[input0, input1], outputs=output)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for input, target in dataset:

    image1 = tf.convert_to_tensor(input[1])

    image0_var = tf.Variable(input[0])

    for iteration in range(2):
        image0 = tf.convert_to_tensor(input[0])
        with tf.GradientTape() as tape:
            tape.watch(image0)
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            # prediction = model(input, training=False)  # Logits for this minibatch
            prediction = model([input[0], input[1]])
            print('prediction: {}'.format(prediction))
            # Compute the loss value for this minibatch.
            # loss_value = loss_fun(target, prediction)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(prediction, [input[0]])
        #print(grads)  # output: [None]
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.

        print('mean before apply_gradients: {}'.format(tf.reduce_mean(image0_var)))
        print('mean gradss: {}'.format(tf.reduce_mean(grads)))
        optimizer.apply_gradients(zip(grads, [image0_var]))
        print('mean after apply_gradients: {}'.format(tf.reduce_mean(image0_var)))
        #optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print('Iteration {}'.format(iteration))
