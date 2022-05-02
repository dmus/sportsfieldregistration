import tensorflow as tf
import matplotlib.pyplot as plt

from sportsfieldregistration.metrics import IOUPart
from sportsfieldregistration.utils import decode_img, get_dataset, visualize, warp_image, get_perspective_transform


def create_errors(images, labels, metric):
    batch_size = tf.shape(images)[0]
    # Add global random translation
    global_translation = tf.random.uniform((batch_size, 2), -0.05, 0.05)
    global_translation = tf.tile(global_translation, [1, 4])
    noisy_labels = labels + global_translation
    #noisy_labels = labels
    # Add local random translation
    local_translation = tf.random.uniform((batch_size, 8), -0.02, 0.02)
    noisy_labels = noisy_labels + local_translation

    errors = metric.get_iou(labels, noisy_labels)
    print(errors)

    ref_corners = tf.constant([[[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]]], dtype=tf.float32)
    #ref_corners = np.array([[[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]]]).astype(np.float32)
    ref_corners = tf.repeat(ref_corners, [batch_size], axis=0)
    template = tf.expand_dims(decode_img('/home/derk/sports-field-registration/data/world_cup_template.png'), axis=0)
    templates = tf.repeat(template, [batch_size], axis=0)
    return (images, noisy_labels, ref_corners, templates), errors

tf.random.set_seed(1234)
dataset_train = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=1)


metric = IOUPart()
for input, target in dataset_train:
    #import pdb;pdb.set_trace()
    inputs, targets = create_errors(input, target, metric)

    template = decode_img('/home/derk/sports-field-registration/data/world_cup_template.png')

    ref_corners = inputs[2]
    H = inputs[1][0]
    H = tf.transpose(tf.reshape(H, (-1, 2, 4)), [0, 2, 1])
    H = get_perspective_transform(ref_corners, H)
    out_shape = (720, 1280, 3)
    warped_template = warp_image(template, H, out_shape=out_shape)

    img = inputs[0][0]

    plt.imshow(img);
    plt.show()
    plt.imshow(tf.squeeze(warped_template));plt.show()
    #visualize(inputs[3][0], inputs[1][0])

