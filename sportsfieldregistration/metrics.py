import tensorflow as tf
import matplotlib.pyplot as plt
from sportsfieldregistration.utils import warp_image, get_perspective_transform


class IOUPart(tf.keras.metrics.Metric):
    def __init__(self, name='iou_part', **kwargs):
        super(IOUPart, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou_part', initializer='zeros')
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')

        self.template_width = 1050
        self.template_height = 680

        self.frame_width = 1280
        self.frame_height = 720

        self.corners = tf.constant([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = self.get_iou(y_true, y_pred)

        self.iou.assign_add(tf.reduce_sum(iou))
        self.num_examples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def get_iou(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        fake_frame = tf.ones((batch_size, self.frame_height, self.frame_width, 1))
        fake_template = tf.zeros((batch_size, self.template_height, self.template_width, 1))

        target_source = tf.repeat(tf.expand_dims(self.corners, axis=0), [batch_size], axis=0)
        target_destination = tf.transpose(tf.reshape(y_true, (-1, 2, 4)), [0, 2, 1])
        H_target = get_perspective_transform(target_source, target_destination)

        output_source = tf.repeat(tf.expand_dims(self.corners, axis=0), [batch_size], axis=0)
        output_destination = tf.transpose(tf.reshape(y_pred, (-1, 2, 4)), [0, 2, 1])

        H_output = get_perspective_transform(output_source, output_destination)
        output_mask = warp_image(fake_frame, tf.linalg.inv(H_output), out_shape=fake_template.shape[-3:-1])
        target_mask = warp_image(fake_frame, tf.linalg.inv(H_target), out_shape=fake_template.shape[-3:-1])

        intersection_mask = output_mask * target_mask
        output = tf.reduce_sum(output_mask, axis=[1, 2, 3])
        target = tf.reduce_sum(target_mask, axis=[1, 2, 3])
        intersection = tf.reduce_sum(intersection_mask, axis=[1, 2, 3])
        union = output + target - intersection
        iou = tf.math.divide_no_nan(intersection, union)

        return iou

    def result(self):
        return self.iou / self.num_examples

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.iou.assign(0.)
        self.num_examples.assign(0.)


class IOUWhole(tf.keras.metrics.Metric):
    def __init__(self, name='iou_whole', **kwargs):
        super(IOUWhole, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou_whole', initializer='zeros')
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')

        self.template_width = 1050
        self.template_height = 680

        self.frame_width = 1280
        self.frame_height = 720

        self.corners = tf.constant([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = self.get_iou(y_true, y_pred)

        self.iou.assign_add(tf.reduce_sum(iou))
        self.num_examples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def get_iou(self, y_true, y_pred):
        ZOOM_OUT_SCALE = 4
        batch_size = tf.shape(y_true)[0]
        fake_template = tf.zeros((batch_size, self.template_height * ZOOM_OUT_SCALE, self.template_width * ZOOM_OUT_SCALE, 1))

        scaling_mat = tf.convert_to_tensor([[ZOOM_OUT_SCALE, 0, 0],
                                            [0, ZOOM_OUT_SCALE, 0],
                                            [0, 0, 1]], dtype=tf.float32)
        scaling_mat = tf.repeat(scaling_mat, repeats=batch_size, axis=0)

        target_mask = warp_image(fake_template, scaling_mat, out_shape=fake_template.shape[-3:-1])

        target_source = tf.repeat(tf.expand_dims(self.corners, axis=0), [batch_size], axis=0)
        target_destination = tf.transpose(tf.reshape(y_true, (-1, 2, 4)), [0, 2, 1])
        H_target = get_perspective_transform(target_source, target_destination)

        output_source = tf.repeat(tf.expand_dims(self.corners, axis=0), [batch_size], axis=0)
        output_destination = tf.transpose(tf.reshape(y_pred, (-1, 2, 4)), [0, 2, 1])
        H_output = get_perspective_transform(output_source, output_destination)

        mapping_mat = tf.matmul(H_output, scaling_mat)
        mapping_mat = tf.matmul(tf.linalg.inv(H_target), mapping_mat)
        output_mask = warp_image(fake_template, mapping_mat, out_shape=fake_template.shape[-3:-1])

        output_mask = tf.cast(output_mask >= 0.5, tf.float32)
        target_mask = tf.cast(target_mask >= 0.5, tf.float32)
        intersection_mask = output_mask * target_mask
        output = tf.reduce_sum(output_mask, axis=[1, 2, 3])
        target = tf.reduce_sum(target_mask, axis=[1, 2, 3])
        intersection = tf.reduce_sum(intersection_mask, axis=[1, 2, 3])
        union = output + target - intersection
        iou = tf.math.divide_no_nan(intersection, union)
        return iou

    def result(self):
        return self.iou / self.num_examples

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.iou.assign(0.)
        self.num_examples.assign(0.)
