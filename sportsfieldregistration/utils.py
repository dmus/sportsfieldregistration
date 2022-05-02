import os
from glob import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def warp_image(img, H, out_shape=None, input_grid=None):
    """Warp img with homography H
    """
    if out_shape is None:
        out_shape = img.shape[-3:-1]
    #out_shape = img.shape[-3:-1]

    if len(img.shape) < 4:
        img = img[None]
    if len(H.shape) < 3:
        H = H[None]

    #assert tf.shape(img)[0] == tf.shape(H)[0], 'batch size of images do not match the batch size of homographies'
    batchsize = tf.shape(img)[0]

    # create grid for interpolation (in frame coordinates)
    if input_grid is None:
        x, y = tf.meshgrid(
            tf.linspace(-0.5, 0.5, num=out_shape[1]),
            tf.linspace(-0.5, 0.5, num=out_shape[0])
        )
    else:
        x, y = input_grid

    x, y = tf.reshape(x, [-1]), tf.reshape(y, [-1])

    # append ones for homogeneous coordinates
    xy = tf.stack([x, y, tf.ones(x.shape)], axis=-1)
    xy = tf.repeat(tf.expand_dims(xy, axis=0), repeats=[batchsize], axis=0)  # shape: (B, 3, N)
    #xy = tf.keras.backend.repeat_elements(tf.expand_dims(xy, axis=0), rep=batchsize, axis=0)

    # warp points to model coordinates
    xy_warped = tf.matmul(H, tf.transpose(xy, [0, 2, 1]))  # H.bmm(xy)
    xy_warped, z_warped = tf.split(xy_warped, [2, 1], axis=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped)
    x_warped, y_warped = tf.unstack(xy_warped, axis=1)

    # build grid
    grid = tf.stack([
        tf.reshape(y_warped, (batchsize, out_shape[0], out_shape[1])),
        tf.reshape(x_warped, (batchsize, out_shape[0], out_shape[1]))
    ], axis=-1)

    warped_img = bilinear_sampler(img, grid[:, :, :, 1], grid[:, :, :, 0])

    # xy_warped = tf.transpose(xy_warped, [0, 2, 1])

    # warped_img = tfa.image.interpolate_bilinear(img, xy_warped, indexing='xy')
    # warped_img = tf.reshape(warped_img, (batchsize, out_shape[-2], out_shape[-1], -1))
    # sample warped image
    # warped_img = torch.nn.functional.grid_sample(
    #    img, grid, mode='bilinear', padding_mode='zeros')

    # if tf.reduce_any(tf.math.is_nan(warped_img)):
    #     print('nan value in warped image! set to zeros')
    #     warped_img[tf.math.is_nan(warped_img)] = 0

    return warped_img


def get_four_corners(homo_mat, canon4pts):
    """
    Calculate the 4 corners after transformation, from frame to template
    assuming the original 4 corners of the frame are [+-0.5, +-0.5]
    note: this function supports batch processing
    Arguments:
        homo_mat {[type]} -- [homography, shape: (B, 3, 3) or (3, 3)]
    Return:
        xy_warped -- torch.Size([B, 2, 4])
    """
    # append ones for homogeneous coordinates
    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    assert homo_mat.shape[1:] == (3, 3)
    # if canon4pts is None:
    #     canon4pts = utils.to_torch(utils.FULL_CANON4PTS_NP())

    # canon4pts = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    # w = 115
    # h = 75
    # canon4pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0,h-1]])

    assert canon4pts.shape == (4, 2)
    x, y = canon4pts[:, 0], canon4pts[:, 1]
    xy = tf.stack([x, y, tf.ones(x.shape)])

    # warp points to model coordinates
    # homo_mat = tf.convert_to_tensor(homo_mat, dtype=tf.float32)
    xy_warped = tf.matmul(homo_mat, xy)

    xy_warped, z_warped = tf.split(xy_warped, [2, 1], axis=1)
    xy_warped = xy_warped / (z_warped + 1e-8)
    return xy_warped


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator. x, y in range [-1,1]
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # To fix issue with NaNs
    out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)

    return out


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """

    # if not src.shape == dst.shape:
    #     raise ValueError("Inputs must have the same shape. Got {} and {}"
    #                      .format(src.shape, dst.shape))
    # if not (src.shape[0] == dst.shape[0]):
    #     raise ValueError("Inputs must have same batch size dimension. Got {} and {}"
    #                      .format(src.shape, dst.shape))

    def ax(p, q):
        ones = tf.ones(tf.shape(p))[..., 0:1]
        zeros = tf.zeros(tf.shape(p))[..., 0:1]
        return tf.concat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], axis=1)

    def ay(p, q):
        ones = tf.ones(tf.shape(p))[..., 0:1]
        zeros = tf.zeros(tf.shape(p))[..., 0:1]
        return tf.concat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], axis=1)

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    p.append(ax(src[:, 0], dst[:, 0]))
    p.append(ay(src[:, 0], dst[:, 0]))

    p.append(ax(src[:, 1], dst[:, 1]))
    p.append(ay(src[:, 1], dst[:, 1]))

    p.append(ax(src[:, 2], dst[:, 2]))
    p.append(ay(src[:, 2], dst[:, 2]))

    p.append(ax(src[:, 3], dst[:, 3]))
    p.append(ay(src[:, 3], dst[:, 3]))

    # A is Bx8x8
    A = tf.stack(p, axis=1)

    # b is a Bx8x1
    b = tf.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], axis=1)

    # solve the system Ax = b
    X = tf.linalg.solve(A, b)  # Bx8x1

    # create variable to return
    batch_size = tf.shape(src)[0]
    ones = tf.ones((batch_size, 1, 1), dtype=src.dtype)
    M = tf.concat([X, ones], axis=1)
    H = tf.reshape(M, (-1, 3, 3))  # Bx3x3

    return H


def decode_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def map(image_file, label):
    img = decode_img(image_file)
    img = tf.image.resize(img, size=(360, 640))
    return img, label


def get_dataset(folder, shuffle=False, batch_size=1):
    label_files = sorted(glob(os.path.join(folder, '*.homographyMatrix')))
    image_files = sorted(glob(os.path.join(folder, '*.jpg')))

    labels = get_labels(label_files)

    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))

    if shuffle:
        dataset = dataset.shuffle(len(image_files))
    dataset = dataset.map(map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def get_labels(label_files):
    """
    Original labels map from 1280x720 to 115x75
    """
    to_unnormalized_space = np.asarray([[1280, 0, 640],
                                        [0, 720, 360],
                                        [0, 0, 1]], dtype=np.float32)

    to_normalized_space = np.asarray([[0.5 / 57.5, 0, -0.5],
                                      [0, 0.5 / 37.5, -0.5],
                                      [0, 0, 1]], dtype=np.float32)

    labels = []
    corners = np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)
    for file in label_files:
        with open(file) as f:
            content = f.readlines()
            H = np.zeros((3, 3), dtype=np.float32)
        for i in range(len(content)):
            H[i] = np.array([float(x) for x in content[i].strip().split()], dtype=np.float32)
        H = tf.linalg.matmul(H, to_unnormalized_space)
        H = tf.linalg.matmul(to_normalized_space, H)

        warped_corners = get_four_corners(H, corners)
        labels.append(np.reshape(warped_corners, (-1,)))

    labels = np.asarray(labels)
    return labels


def visualize(image, points):
    template_path = '/home/derk/sports-field-registration/data/world_cup_template.png'
    template_img = tf.io.read_file(template_path)
    # template_img = tf.image.decode_png(template_img, channels=3)
    template_img = tf.image.decode_jpeg(template_img, channels=3)  # convert the compressed string to a 3D uint8 tensor
    template_img = tf.image.convert_image_dtype(template_img, tf.float32)

    # Show data
    corners = np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)

    warped_corners = tf.reshape(points, (-1, 2, 4))
    H = get_perspective_transform(tf.expand_dims(corners, axis=0), tf.transpose(warped_corners, [0, 2, 1]))
    warped_corners = get_four_corners(H, corners)

    map = np.array([[1050, 0, 525],
                    [0, 680, 340],
                    [0, 0, 1]], dtype=np.float32)

    x, y = tf.squeeze(warped_corners[:, 0]), tf.squeeze(warped_corners[:, 1])
    xy = tf.stack([x, y, tf.ones(x.shape)])
    warped_corners = tf.matmul(map, xy)
    warped_corners = warped_corners[0:2, :]

    warped_image = warp_image(image, tf.linalg.inv(H), out_shape=template_img.shape[:2])
    warped_image = warped_image[0]

    show_image = np.copy(template_img)
    valid_index = warped_image[:, :, 0] > 0.0
    overlay = (template_img[valid_index] + warped_image[valid_index]) / 2
    show_image[valid_index] = overlay
    plt.imshow(show_image)
    plt.scatter(warped_corners[0, :], warped_corners[1, :])
    plt.show()


def random_flip(images, labels):
    batch_size = tf.shape(images)[0]
    flipped_images = tf.image.flip_left_right(images)

    points = tf.reshape(labels, (-1, 2, 4))
    x_flipped = points[:, 0, :] * -1
    y = points[:, 1, :]

    flipped_labels = tf.stack([x_flipped, y], axis=1)
    flipped_labels = tf.gather(flipped_labels, [3, 2, 1, 0], axis=2)
    flipped_labels = tf.reshape(flipped_labels, (batch_size, -1))

    which = tf.random.uniform((batch_size,)) > 0.5
    images = tf.where(tf.reshape(which, (batch_size, 1, 1, 1)), flipped_images, images)
    labels = tf.where(tf.reshape(which, (batch_size, 1)), flipped_labels, labels)
    return images, labels


def create_errors(images, labels, metric):
    batch_size = tf.shape(images)[0]

    tf.random.set_seed(1234)

    # Add global random translation
    global_translation = tf.random.uniform((batch_size, 2), -0.05, 0.05)
    global_translation = tf.tile(global_translation, [1, 4])
    noisy_labels = labels + global_translation

    # Add local random translation
    local_translation = tf.random.uniform((batch_size, 8), -0.02, 0.02)
    noisy_labels = noisy_labels + local_translation

    errors = metric.get_iou(labels, noisy_labels)

    ref_corners = tf.constant([[[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]]], dtype=tf.float32)
    ref_corners = tf.repeat(ref_corners, [batch_size], axis=0)

    template = tf.expand_dims(decode_img('/home/derk/sports-field-registration/data/world_cup_template.png'), axis=0)
    templates = tf.repeat(template, [batch_size], axis=0)

    return (images, noisy_labels, ref_corners, templates), errors
