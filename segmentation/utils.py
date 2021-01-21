"""
Utils functions.
"""

from scipy.ndimage import distance_transform_edt as distance
from cv2 import findContours
from cv2 import RETR_EXTERNAL, CHAIN_APPROX_NONE, cvtColor, COLOR_BGR2GRAY
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Conv3DTranspose
from keras.layers.pooling import AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D, MaxPool3D
from keras.layers import Input, Concatenate, Lambda, Dropout, Concatenate, Multiply, Softmax, Reshape, UpSampling3D, \
    Subtract, Add, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def calc_SDM(seg):
    """
    Computes Signed Distance Map of input ground truth image or volume using scipy function.
    In case seg is 3D volume, it separately computes 2D SDM fo each single slice.
    Args:
        seg: 2D or 3D binary array to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if len(seg.shape) == 2:
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return res
    elif len(seg.shape) == 3:
        for i in range(seg.shape[2]):
            pos = posmask[:, :, i]
            if pos.any():
                neg = ~pos
                res[:, :, i] = distance(neg) * neg - (distance(pos) - 1) * pos
        return res
    else:
        print("Could not recognise dimensions")


def calc_SDM_batch(y_true, numClasses):
    """
    Prepares the input for distance maps computation, and pass it to calc_SDM
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_SDM(y)
    return np.array(dist_batch).astype(np.float32)


def calc_DM(seg):
    """
    Computes Non-Signed Distance Map of input ground truth image or volume using scipy function.
    In case seg is 3D volume, it separately computes 2D DM fo each single slice.
    Args:
        seg: 2D or 3D binary array to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if len(seg.shape) == 2:
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask + (distance(posmask) - 1) * posmask
        return res
    elif len(seg.shape) == 3:
        for i in range(seg.shape[2]):
            pos = posmask[:, :, i]
            if pos.any():
                neg = ~pos
                res[:, :, i] = distance(neg) * neg + (distance(pos) - 1) * pos
        return res
    else:
        print("Could not recognise dimensions")


def calc_DM_batch(y_true, numClasses):
    """
    Prepares the input for distance maps computation, and pass it to calc_DM
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_DM(y)
    return np.array(dist_batch).astype(np.float32)


def calc_DM_edge(seg):
    """
    Computes Non-Signed Distance Map of input ground-truth image/volume CONTOURS using scipy function.
    In case seg is 3D volume, it separately computes 2D DM fo each single slice.
    Args:
        seg: 2D or 3D binary image of ground truth contours to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    for i in range(seg.shape[2]):
        pos = posmask[:, :, i]
        if pos.any():
            neg = ~pos
            res[:, :, i] = distance(neg)
    return res


def calc_DM_batch_edge(y_true, numClasses):
    """
    Prepares the input (contours) for distance maps computation, and pass it to calc_dist_map.
    To use when loading ground truth contours from disk.
    Args:
        y_true: ground truth contour tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_DM_edge(y)
    return np.array(dist_batch).astype(np.float32)


def calc_DM_batch_edge2(y_true, numClasses):
    """
    Prepares the input for distance maps computation: it takes y_true masks, creates y_true contours
    and passes it to calc_DM_edge.
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
        array of ground truth contours of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    # surface_label = np.zeros((numClasses - 1, ) + y_true_numpy.shape[1::])
    # dist_batch = np.zeros((numClasses - 1, ) + y_true_numpy.shape[1::])
    surface_label = np.zeros_like(y_true_numpy)
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(1, numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            for k in range(y.shape[2]):
                img_lab = y[:, :, k].astype(np.uint8)
                contour_lab, hierarchy_lab = findContours(img_lab, RETR_EXTERNAL, CHAIN_APPROX_NONE)
                if len(contour_lab) != 0:  # CONTOUR PER SLICE IS PRESENT
                    for j in range(len(contour_lab)):
                        if contour_lab[j].shape[1] == 1:
                            contour_lab[j].resize(contour_lab[j].shape[0], 2)
                        surface_label[c, i, contour_lab[j][:, 1], contour_lab[j][:, 0], k] = 1
                else:
                    surface_label[c, i, :, :, k] = np.zeros_like(img_lab)
            dist_batch[c, i] = calc_DM_edge(surface_label[c, i])

    surface_label[0] = surface_label[1] + surface_label[2] + surface_label[3] + surface_label[4]
    for i in range(y_true_numpy.shape[1]):
        dist_batch[0,i] = calc_DM_edge(surface_label[0, i])
    surface_label[0] = 1 - surface_label[0]
    return np.array(dist_batch).astype(np.float32), np.array(surface_label).astype(np.float32)


def computeContours(y_true, numClasses):
    """
    Receive y_true masks and creates y_true contours
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes

    Returns: array of ground truth contours of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    surface_label = np.zeros((numClasses - 1, ) + y_true_numpy.shape[1::])
    for c in range(1, numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            for k in range(y.shape[2]):
                img_lab = y[:, :, k].astype(np.uint8)
                contour_lab, hierarchy_lab = findContours(img_lab, RETR_EXTERNAL, CHAIN_APPROX_NONE)
                if len(contour_lab) != 0:  # CONTOUR PER SLICE IS PRESENT
                    for j in range(len(contour_lab)):
                        if contour_lab[j].shape[1] == 1:
                            contour_lab[j].resize(contour_lab[j].shape[0], 2)
                        surface_label[c-1, i, contour_lab[j][:, 1], contour_lab[j][:, 0], k] = 1
                else:
                    surface_label[c-1, i, :, :, k] = np.zeros_like(img_lab)
    # surface_label[0] = surface_label[1] + surface_label[2] + surface_label[3] + surface_label[4]
    return np.array(surface_label).astype(np.float32)


def count_class_voxels(labels, nVoxels, numClasses):
    """
    Counts total number of voxels for each class in the batch size.
    input is supposed to be 4 or 5-dimensional: (class, batch, rows, columns) or
    (class, batch, rows, columns, slices).
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        out: list with number of voxel per class
    """
    out = [0] * numClasses
    out[0] = 0
    for c in range(1, numClasses):
        out[c] = tf.math.count_nonzero(labels[c])
    first_term = tf.cast(nVoxels, 'int64')
    second_term = tf.reduce_sum(out)
    out[0] = tf.subtract(first_term, second_term)
    return out


def get_loss_weights(labels, nVoxels, numClasses):
    """
    Compute loss weights for each class.
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        1D tf.tensor of len = numClasses containing weights for each class
    """

    numerator_1 = count_class_voxels(labels, nVoxels, numClasses)
    numerator_1 = tf.math.sqrt(tf.divide(1.0, numerator_1))
    numerator_1 = tf.divide(numerator_1, tf.reduce_sum(numerator_1))

    return numerator_1


# Functions for CDDUnet and CDUnet

def conv_factory(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4, kernel_size = (3,3,3)):
    """
    This function defines the convolution operation to perform in each layer of a dense block
    Args:
        x: Input tensor
        concat_axis: axis of concatenation
        nb_filter: number of features of input tensor
        dropout_rate: probability of dropout layers
        weight_decay: weight decay parameter
        kernel_size: kernel size used for convolution. Default (3,3,3)

    Returns: Tensor to pass to the next layer
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(4*nb_filter, (1, 1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               # use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(nb_filter, kernel_size,
               kernel_initializer="he_uniform",
               padding="same",
               # use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, nb_filter, theta, dropout_rate=None, weight_decay=1E-4):
    """
    Transition layer after each dense block.
    It reduces tensor dimension and number of features.
    Args:
        x: Input tensor
        concat_axis: axis of concatenation
        nb_filter: number of features of input tensor
        theta: parameter in (0,1] to specify number of features in output.
            features_out = theta * features_in
        dropout_rate: probability of dropout layers
        weight_decay: weight decay parameter

    Returns: returns resized tensor in order to reduce dimensionality
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter * theta, (1, 1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               # use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x, nb_filter * theta


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """
    Create a dense connected block of depth nb_layers,
    where each output is fed into all subsequent layers.
    Args:
        x: tensor in input
        concat_axis: axis of concatenation
        nb_layers: number of layers of the dense block
        nb_filter: number of filters of input tensor
        growth_rate: number of output features for each layer in the block
        dropout_rate: probability of dropout layers
        weight_decay: weight decay parameter

    Returns: Output tensor with same shape and number of features
    equal to: nb_filter + nb_layers * growth_rate
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter


def channelModule(input_tensor, nb_filter):
    """
    Channel contextual model to enhance channel information flow.
    Args:
        input_tensor: input tensor
        nb_filter: number of features of the input tensor
    Returns: tensor of the same shape of input where relevant channel have been enhanced
        in contrast to less relevant ones.
    """

    scale_tensor = tf.ones_like(input_tensor)

    x = GlobalAveragePooling3D()(input_tensor)
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)
    x = Conv1D(nb_filter/2, 1,
               kernel_initializer="he_uniform")(x)
    x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
               kernel_initializer="he_uniform")(x)
    x = Activation('sigmoid')(x)

    x = tf.multiply(x, scale_tensor)  # Lambda(lambda y: tf.multiply(y, scale_tensor))(x)
    output_tensor = Multiply()([x, input_tensor])

    return output_tensor


def spatialModule(input_tensor, nb_filter):
    """
    Spatial contextual model to enhance spatial information across features.
    Args:
        input_tensor: input tensor
        nb_filter: number of features in the input tensor.

    Returns: tensor of the same shape of input where relevant patches inside the
        volume are enhanced in contrast to less relebant ones.
    """

    scale_tensor = tf.ones_like(input_tensor)

    x = Conv3D(nb_filter/2, (1, 1, 1),
               kernel_initializer="he_uniform")(input_tensor)
    x = Activation('relu')(x)
    x = Conv3D(1, (1, 1, 1),
               kernel_initializer="he_uniform")(x)
    x = Activation('sigmoid')(x)

    x = tf.multiply(x, scale_tensor)  # Lambda(lambda y: tf.multiply(y, scale_tensor))(x)
    output_tensor = Multiply()([x, input_tensor])
    return output_tensor


def denseUnit(input_tensor, dense_filters=48, weight_decay=1E-4, kernel_size = (3,3,3)):
    """
    Dense unit that comprehends a convolution with a fixed number of filters and the two
        spatial and channel contextual modules.
    Args:
        input_tensor: input tensor
        dense_filters: number of filters for the first convolution
        weight_decay: weight decay parameter
        kernel_size: kernel used for convolution

    Returns: tensor with the same shape as input
    """

    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(input_tensor)
    x = Activation('relu')(x)
    x = Conv3D(dense_filters, kernel_size, padding='same')(x)
    x = Dropout(0.2)(x)

    x = spatialModule(x, dense_filters)
    x = channelModule(x, dense_filters)

    x = Concatenate()([x, input_tensor])
    return x


def compressionUnit(input_tensor, nb_filter, kernel_size=(3,3,3)):
    """
    Compression unit to reduce the linear increase of feature numbers in the dense units.
    Args:
        input_tensor: input tensor
        nb_filter: number fo filters for the convolution
        kernel_size: kernel size used for convolution. Default (3,3,3)

    Returns: a tensor with number of features specified in nb_filter
    """

    x = Conv3D(nb_filter, kernel_size, padding='same')(input_tensor)

    x = spatialModule(x, nb_filter)
    x = channelModule(x, nb_filter)
    return x


def upsamplingUnit(encoding_input, decoding_input, filter_enc, filter_dec, kernel_size=(3,3,3),
                   deconv_kernel_size=(2,2,2), deconv_strides=(2, 2, 2)):
    """
    Upsampling unit that upsamples from decoding path and concatenates with encoding path to
        maintain fine spatial information and produce dense predictions.
    Args:
        encoding_input: input tensor coming from encoding path
        decoding_input: input tensor coming from decoding path
        filter_enc: number fo filter for the convolution of the encoding path
        filter_dec: number fo filter for the transposed convolution of the decoding path
        kernel_size: kernel size used for convolution. Default (3,3,3)
        deconv_kernel_size: kernel size used for deconvolution. Default (2,2,2)
        deconv_strides: strides used for deconvolution. Default (2,2,2)

    Returns: concatenation of the two tensors in input after convolutions
    """

    x = Conv3D(filter_enc, kernel_size, padding='same')(encoding_input)
    y = Conv3DTranspose(filter_dec, deconv_kernel_size, strides=deconv_strides, padding='same')(decoding_input)
    return Concatenate()([x, y])


############################## ––––––––––––––––– ##############################
# Functions for DoubleUnet
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = inputs

    x = Conv3D(filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def encoder1(inputs):
    num_filters = [8, 16, 32, 64]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool3D((2, 2, 2))(x)

    return x, skip_connections


def decoder1(inputs, skip_connections):
    num_filters = [64, 32, 16, 8]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling3D((2, 2, 2))(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    return x


def encoder2(inputs):
    num_filters = [8, 16, 32, 64]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool3D((2, 2, 2))(x)

    return x, skip_connections


def decoder2(inputs, skip_1, skip_2):
    num_filters = [64, 32, 16, 8]
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling3D((2, 2, 2))(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x


def output_block(inputs):
    x = Conv3D(8, (1, 1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x


def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)

############################## ––––––––––––––––– ##############################


# Functions for Unet2
def PEE(x, filters):
    if filters > 30:
        pool_size_1 = (3, 3, 3)
        pool_size_2 = (5, 5, 5)
    else:
        pool_size_1 = (5, 5, 5)
        pool_size_2 = (7, 7, 7)

    x = Conv3D(filters/2, (1, 1, 1), padding='same')(x)
    x_1 = AveragePooling3D(pool_size=pool_size_1, strides = (1, 1, 1), padding='same')(x)
    x_2 = AveragePooling3D(pool_size=pool_size_2, strides = (1, 1, 1), padding='same')(x)

    x_11 = Subtract()([x, x_1])
    x_22 = Subtract()([x, x_2])

    x = Concatenate()([x, x_11, x_22])
    x = Conv3D(filters, (1, 1, 1), padding='same')(x)
    return x


def RA(upsampled, high_level, filters):
    x = Activation('sigmoid')(upsampled)
    x = -1 * x + 1
    x = Multiply()([x, high_level])

    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = Add()([x, upsampled])
    return x


def MINI_MTL(inputs, filters, numClasses, i):
    x_edge = RA(inputs, inputs, filters)
    x_mask = RA(inputs, inputs, filters)

    x_edge = Conv3D(filters, (3, 3, 3), padding='same')(x_edge)
    x_edge = BatchNormalization(axis=-1)(x_edge)
    x_edge = Activation('relu')(x_edge)
    x_mask = Conv3D(filters, (3, 3, 3), padding='same')(x_mask)
    x_mask = BatchNormalization(axis=-1)(x_mask)
    x_mask = Activation('relu')(x_mask)

    out_edge = Conv3D(numClasses - 1, (1, 1, 1), padding='same')(x_edge)
    out_edge = UpSampling3D(pow(2,i))(out_edge)
    out_edge = Softmax(axis=-1, dtype='float32', name='out_edge_{}'.format(i))(out_edge)
    out_mask = Conv3D(numClasses, (1, 1, 1), padding='same')(x_mask)
    out_mask = UpSampling3D(pow(2,i))(out_mask)
    out_mask = Softmax(axis=-1, dtype='float32', name='out_mask_{}'.format(i))(out_mask)

    # out_mtl = Add()([x_mask, x_edge])
    out_mtl = Concatenate()([x_mask, x_edge])
    out_mtl = Conv3D(filters, (1, 1, 1), padding='same')(out_mtl)

    return out_mtl, out_edge, out_mask


def build_MINI_MTL(input_shape, filters, numClasses, i):
    input_layer = Input(shape=(input_shape, input_shape, input_shape, filters))
    x_edge = RA(input_layer, input_layer, filters)
    x_mask = RA(input_layer, input_layer, filters)

    x_edge = Conv3D(filters, (3, 3, 3), padding='same')(x_edge)
    x_edge = BatchNormalization(axis=-1)(x_edge)
    x_edge = Activation('relu')(x_edge)
    x_mask = Conv3D(filters, (3, 3, 3), padding='same')(x_mask)
    x_mask = BatchNormalization(axis=-1)(x_mask)
    x_mask = Activation('relu')(x_mask)

    out_edge = Conv3D(numClasses, (1, 1, 1), padding='same')(x_edge)
    out_edge = Softmax(axis=-1)(out_edge)
    out_edge = UpSampling3D(pow(2,i), name='out_edge_{}'.format(i))(out_edge)
    out_mask = Conv3D(numClasses, (1, 1, 1), padding='same')(x_mask)
    out_mask = Softmax(axis=-1)(out_mask)
    out_mask = UpSampling3D(pow(2,i), name='out_mask_{}'.format(i))(out_mask)

    out_mtl = Concatenate()([x_mask, x_edge])
    out_mtl = Conv3D(filters, (1, 1, 1), padding='same')(out_mtl)

    mtl_model = Model(inputs=[input_layer], outputs=[out_edge, out_mask])

    return mtl_model, out_mtl


def CFF(input_list, input_size, filters, i):
    out_shape = input_size/pow(2,i)

    y = tf.zeros_like(input_list[i-1])
    for j,x in enumerate(input_list):
        if j < i-1:
            down_factor = int((input_size/pow(2,j+1)) / out_shape)
            x = AveragePooling3D((down_factor, down_factor, down_factor))(x)
            x = Conv3D(filters, (1, 1, 1), padding='same')(x)
            sigm = Activation('sigmoid')(x)
            x = Multiply()([x, sigm])
            y = Add()([y, x])
        if j > i-1:
            up_factor = int(out_shape / (input_size/pow(2,j+1)))
            x = Conv3D(filters, (1, 1, 1), padding='same')(x)
            x = UpSampling3D((up_factor, up_factor, up_factor))(x)
            sigm = Activation('sigmoid')(x)
            x = Multiply()([x, sigm])
            y = Add()([y,x])

    x_i = input_list[i-1]
    x_i_sigm = Activation('sigmoid')(x_i)
    x_i_sigm = -1 * x_i_sigm + 1
    out = Multiply()([x_i_sigm, y])
    out = Add()([out, x_i])
    return out


def ASPP(x, filters):
    shape = x.shape

    y1 = AveragePooling3D(pool_size=(shape[1], shape[2], shape[3]))(x)
    y1 = Conv3D(filters/2, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling3D((shape[1], shape[2], shape[3]))(y1)

    y2 = Conv3D(filters/2, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv3D(filters/2, 3, dilation_rate=2, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv3D(filters/2, 3, dilation_rate=4, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv3D(filters/2, 3, dilation_rate=8, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv3D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y
