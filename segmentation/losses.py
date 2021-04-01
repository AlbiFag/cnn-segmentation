"""
The losses submodule implements loss function to be used
in segmentation tasks.
"""

from segmentation.metrics import MeanDice
from segmentation.utils import calc_SDM_batch, calc_DM_batch, \
     calc_DM_batch_edge, computeContours
from segmentation.utils import count_class_voxels, get_loss_weights
from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config


# 0
def Distanced_CE_Loss(numClasses, alpha):
    """
    Computes Cross Entropy weighted by an exponential transformation of the Euclidean
    Distance Map, called Distance Weighted Map (DWM). Voxels closer to boundaries are weighted more.
    Distanced Cross Entropy is combined with Dice Loss with parameter alpha.

    Distanced_CE is defined as follows:
    .. math::

        L_{Dce} = -\\frac{1}{N} \sum^{N} (\sum_{c=1}^{C} DWM_c \cdot y_c \cdot \log{\hat{y}_c})

    with the DWM defined as:
    .. math::

        DWM_c = 1 + \gamma * exp(-\\frac{EDT_c}{\sigma}), \hspace{3mm} \gamma = 8, \hspace{3mm} \sigma = 10

    C is the number of classes, y is the ground truth, y(hat) is the segmentation map, EDT is the
    Euclidean Distance Transform and N is the total number of voxels.

    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss

    Returns:
        categorical_cross_entropy function
    """

    def dice_distCE(y_true, y_pred):
        """
        Computes Cross Entropy weighted by an exponential transformation of the Euclidean
        Distance Map, called Distance Weighted Map (DWM). Voxels closer to boundaries are weighted more.
        Distanced Cross Entropy is combined with Dice Loss with parameter alpha.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            Balanced combination of Distanced CrossEntropy and Dice
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                                   tf.divide(numerator, denominator)))

        SDM = tf.py_function(func=calc_DM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)

        epsilon = backend_config.epsilon
        gamma = 8
        sigma = 10

        # Exponential transformation of the Distance transform
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM)/sigma)
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(DWM * y_true * math_ops.log(y_pred))/tf.cast(nVoxels, tf.float32)
        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * wcc_loss

    return dice_distCE


# 2
def Weighted_DiceBoundary_Loss(numClasses, alpha):
    """
    Compute multiclass weighted dice index, weighted by the Euclidean Distance Map. Voxels
    further from the boundaries are weighted more.
    Weighted_DiceBoundary_Loss is defined as follows:
    ..math::
        L_{Dce} = -\\frac{1}{N} \sum^{N} (\sum_{c=1}^{C} EDT_c \cdot y_c \cdot \log{\hat{y}_c})

    where EDT is the signed Euclidean Distance Transform, C is the number of classes, y is the ground truth,
    y(hat) is the segmentation map and N is the total number of voxels.

    Args:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and boundary loss

    Returns: multiclass_3D_weighted_dice_boundary_loss
    """

    def multiclass_weighted_dice_boundary_loss(y_true, y_pred):
        """
        Compute multiclass weighted dice index, weighted by the Euclidean Distance Map. Voxels
        further from the boundaries are weighted more.
        Args:
            y_true: ground truth tensor [batch, rows, columns, slices, classes], or [batch, rows, columns, classes]
            y_pred: softmax probabilities predicting classes. Shape must be the same as y_true.
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis = axisSum), tf.reduce_sum(y_pred_c, axis = axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator, denominator)))


        SDM = tf.py_function(func=calc_SDM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)

        boundary_loss = tf.multiply(tf.reduce_sum(tf.multiply(SDM, y_pred)), 1.0/nVoxels)

        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * boundary_loss

    return multiclass_weighted_dice_boundary_loss


# 3
def Focal_Loss(numClasses, alpha):
    """
    Computes Cross Entropy weighted with focal method.
    Voxel classified with less confidence are weighted more in the function.
    Focal Loss is defined as follows:
    .. math::
        L_{focal} = -\\frac{1}{N} \sum_{c=1}^{C} (\sum^{N}(1-\hat{y}_{c})^{\gamma}\cdot y_{c} \cdot \log{\hat{y}_{c}})
    .. math::
        \gamma = 2

    C is the number of classes, y is the ground truth, y(hat) is the segmentation map and N is the total number of voxels.

    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss

    Returns:
        dice_focal function
    """

    def dice_focal(y_true, y_pred):
        """
        Computes Cross Entropy weighted with focal method.
        Voxel classified with less confidence are weighted more in the function.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            Balanced combination of Focal Categorical Crossentropy and Dice
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                                   tf.divide(numerator, denominator)))


        epsilon = backend_config.epsilon
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        focal_loss = -math_ops.reduce_sum(tf.math.square(1 - y_pred) * y_true * math_ops.log(y_pred))/tf.cast(nVoxels, tf.float32)

        return alpha * tf.subtract(1.0, mean_over_classes) + (1-alpha) * focal_loss

    return dice_focal


# 5
def MeanDice_Loss(numClasses):
    """
    Computed mean dice coefficient between probability output mask and ground tuth labels. Class predictions
    are weighted to account for class imbalance.

    MeanDice is defined as follows:
    .. math::
        D(y,\hat{y}) =\sum_{c=1}^{C}w_{c}(\\frac{2\cdot\sum^{N} {y_{c}\cdot\hat{y}_{c}}}{\sum^{N}{y_{c}\cdot{y_{c}}} + \sum^{N}{\hat{y}_{c}}\cdot\hat{y}_{c} })

    where w_c is the weight factor for class c defined as:
    .. math::
        w_c = \\frac{\sqrt{\\frac{1}{N_c}}}{\sum_{i=1}^{C}\sqrt{\\frac{1}{N_i}}}

    C is the number of classes, y is the ground truth, y(hat) is the segmentation map and N is the total number of voxels.

    Args:
        numClasses: number of classes
    Returns:
        mean dice weigthed by class

    """
    def mean_dice(y_true, y_pred):
        """
        Computed mean dice coefficient between probability output mask and ground tuth labels.
        Args:
            y_true:
            y_pred:

        Returns:
            mean dice weigthed by class

        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis = axisSum), tf.reduce_sum(y_pred_c, axis = axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator, denominator)))

        return tf.subtract(1.0, mean_over_classes)

    return mean_dice


# 5
def JaccardContour_Loss(numClasses):
    """
    Computed mean jaccard coefficient between probability output mask and ground truth cotour labels.
    Args:
        numClasses: number of classes

    Returns:
        mean jaccard weigthed by class

    """
    def mean_jaccard(y_true, y_pred):
        """
        Computed mean jaccard coefficient between probability output mask and ground truth cotour labels.
        Args:
            y_true:
            y_pred:

        Returns:
            mean dice weigthed by class

        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        contours, nEdgeVoxels = tf.py_function(func=computeContours,
                                  inp=[y_true, numClasses],
                                  Tout=[tf.float32, tf.float32])

        mean_over_classes = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(contours, nVoxels, numClasses)
        # Loop over each class
        for c in range(numClasses):
            y_true_c = contours[c]
            y_pred_c = y_pred[c]
            numerator = tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis = axisSum)
            denominator = tf.subtract(
                tf.add(tf.reduce_sum(y_true_c, axis = axisSum), tf.reduce_sum(y_pred_c, axis = axisSum)),
                tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            class_loss_weight = loss_weights[c]

            mean_over_classes = tf.add(mean_over_classes,
                                       tf.multiply(class_loss_weight,
                                       tf.divide(numerator, denominator)))

        return tf.subtract(1.0, mean_over_classes)

    return mean_jaccard


# 6
def ExpLog_Loss(numClasses, gamma=1):
    """
    Computes Categorical Cross Entropy and Dice with exponential logarithmic transformations.

    ExpLog Loss is defined as follows:
    .. math::
        L_{D} = \\frac{1}{C} \sum_{c=1}^{C} (-\log{\mathcal{D}_{c}(y,\hat{y})})^{{\gamma}_D}
    .. math::
        L_{CE} = \\frac{1}{C} \sum_{c=1}^{C} w_c(-\\frac{1}{N} \sum^{N} (y_{c} \cdot \log{\hat{y}_{c}})^{{\gamma}_{CE}})
    .. math::
        L_{out} = \\beta \cdot L_{D} + (1-\\beta) \cdot L_{CE}
    .. math::
        \\beta = 0.8 , \hspace{3mm}, {\gamma}_D = 1, \hspace{3mm}, {\gamma}_{CE} = 1

    C is the number of classes, y is the ground truth, y(hat) is the segmentation map and N is the total number of voxels.

    Arguments:
        numClasses: number of classes
        alpha: parameter to weight contribution of dice and distance-weighted categorical crossentropy loss
        gamma: exponential of logaritmic dice and CE

    Returns:
        categorical_cross_entropy function
    """

    def exp_log(y_true, y_pred):
        """
        Computes Categorical Cross Entropy and Dice with exponential logarithmic transformations.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            Balanced combination of explog Crossentropy and explog Dice
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = ops.convert_to_tensor_v2(y_true)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        # gamma = ops.convert_to_tensor_v2(gamma)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # gamma = tf.cast(gamma, tf.float32)

        nVoxels = tf.size(y_true)/numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        add_dice = tf.zeros((1,))
        # Get loss weights
        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)

        epsilon = backend_config.epsilon
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, axis=0, keepdims=True)
        # Compute cross entropy from probabilities.
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)

        # Loop over each class to compute dice coefficient
        for c in range(numClasses):
            y_true_c = y_true[c]
            y_pred_c = y_pred[c]
            numerator = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true_c, y_pred_c), axis=axisSum))
            denominator = tf.add(tf.reduce_sum(y_true_c, axis=axisSum), tf.reduce_sum(y_pred_c, axis=axisSum))
            # class_loss_weight = loss_weights[c]

            add_dice = tf.add(add_dice,
                              tf.math.pow(-math_ops.log(clip_ops.clip_by_value(
                                  tf.divide(numerator, denominator), epsilon_, 1. - epsilon_)), gamma))

        dice_loss = tf.divide(add_dice, numClasses)

        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        wcc_loss = -math_ops.reduce_sum(tf.math.pow(y_true * math_ops.log(y_pred), gamma),
                                        axis=(1, 2, 3, 4)) / tf.cast(nVoxels,tf.float32)
        wcc_loss = tf.reduce_mean(tf.multiply(loss_weights, wcc_loss))

        return 0.8 * dice_loss + 0.2 * wcc_loss

    return exp_log


# 7
def Boundary_CE_Loss(numClasses):
    """
    Computes "double-faced" boundary cross entropy, after the generation of contours ground truth labels with function
    "computeContours".
    Definition is the same as the Region_CE_Loss, but with ground truth and prediction contours, instead of filled masks.
    Args:
        numClasses: number of classes
    Returns:
        boundary_crossentropy function
    """

    def boundary_crossentropy(y_true, y_pred):
        """
        Computes "double-faced" boundary cross entropy, after the generation of contours ground truth labels with function
        "computeContours".
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            value of "double-faced" cross entropy

        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

            # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        contours, nEdgeVoxels = tf.py_function(func=computeContours,
                                  inp=[y_true, numClasses],
                                  Tout=[tf.float32, tf.float32])

        nEdgeVoxels = tf.cast(nEdgeVoxels, tf.float32)
        beta = 1 - nEdgeVoxels / nVoxels
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = beta * tf.multiply(contours, math_ops.log(y_pred))
        second_term = (1 - beta) * tf.multiply((1 - contours), math_ops.log(1 - y_pred))

        bc = - tf.reduce_sum(tf.add(first_term, second_term)) / nEdgeVoxels

        return bc

    return boundary_crossentropy


def Distanced_Boundary_CE_Loss(numClasses):
    """
    Computes distanced "double-faced" boundary cross entropy, after the generation of Ground Truth Contours
    and Euclidean Distance Map map with function "calc_DM_batch_edge".
    Definition is the same as the Distanced_Region_CE_Loss, but with ground truth and prediction contours, instead of filled masks.

    Args:
        numClasses: number of classes
    Returns: dist_boundary_crossentropy function

    """

    def dist_boundary_crossentropy(y_true, y_pred):
        """
        Computes distanced "double-faced" boundary cross entropy, after the generation of Ground Truth Contours
        and Euclidean Distance Map map with function "calc_DM_batch_edge".
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true

        Returns:
            Value of distanced "double-faced" cross entropy

        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

            # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        SDM, contours = tf.py_function(func=calc_DM_batch_edge,
                                  inp=[y_true, numClasses],
                                  Tout=[tf.float32, tf.float32])

        nEdgeVoxels = tf.math.count_nonzero(contours)
        nEdgeVoxels = tf.cast(nEdgeVoxels, tf.float32)

        gamma = 8
        sigma = 10
        # Exponential transformation of the Distance transform
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM) / sigma)

        beta = 1 - nEdgeVoxels / nVoxels
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = beta * tf.multiply(DWM, tf.multiply(contours, math_ops.log(y_pred)))
        second_term = (1 - beta) * tf.multiply(DWM, tf.multiply((1 - contours), math_ops.log(1 - y_pred)))

        bc = - tf.reduce_sum(tf.add(first_term, second_term)) / nEdgeVoxels

        return bc

    return dist_boundary_crossentropy


# 8
def Region_CE_Loss(numClasses):
    """
    Computes the "double-faced" regional cross entropy function.
    Region_CE_Loss is defined as follows:
    .. math::
        L_{CE1}^{(c)} = - \\frac{1}{N} \sum^{N} (y_{c} \cdot \log{\hat{y}_{c}})
    .. math::
        L_{CE2}^{(c)} = - \\frac{1}{N} \sum^{N} ((1-y_{c}) \cdot \log{(1-\hat{y}_{c})})
    .. math::
        L_{out} = \sum_{c=1}^{C} w_c * (0.5 \cdot L_{CE1}^{(c)} + 0.5 \cdot L_{CE2}^{(c)})

    C is the number of classes, y is the ground truth, y(hat) is the segmentation map, N is the total number of voxels
    and w_c is the weight assigned to each class to account for class imbalance.

    Args:
        numClasses: number of classes
    Returns:
        region_crossentropy_loss function

    """

    def region_crossentropy_loss(y_true, y_pred):
        """
        Computes the "double-faced" regional cross entropy function.
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
        value of the the "double-faced" regional cross entropy
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = 0.5 * tf.multiply(y_true, math_ops.log(y_pred))
        second_term = 0.5 * tf.multiply((1 - y_true), math_ops.log(1 - y_pred))

        bc_temp = - tf.reduce_sum(tf.add(first_term, second_term), axis=(1, 2, 3, 4))
        bc = tf.reduce_sum(tf.multiply(loss_weights, bc_temp)) / nVoxels

        return bc

    return region_crossentropy_loss


def Distanced_Region_CE_Loss(numClasses):
    """
    Computes distanced "double-faced" regional cross entropy weighted with Euclidean Distance Map,
    after generation of Euclidean Distance Map with function "calc_DM_batch".
    Distanced_region_CE_Loss is defined as follows:
    .. math::
        L_{1}^{(c)} = \\frac{1}{N} \sum^{N} (-DWM \cdot y_{c} \cdot \log{\hat{y_{c}}})
    .. math::
        L_{2}^{(c)} = \\frac{1}{N} \sum^{N} (-DWM \cdot (1-y_{c}) \cdot \log{(1-\hat{y_{c}})})
    .. math::
        L_{edge} = \sum_{c=1}^{C} w_c * [\\beta \cdot L_{1}^{(c)} + (1 - \\beta) * L_{2}^{(c)}]

    with the DWM defined as:
    .. math::
        DWM_c = 1 + \gamma * exp(-\\frac{EDT_c}{\sigma}), \hspace{3mm} \gamma = 8, \hspace{3mm} \sigma = 10

    C is the number of classes, EDT is the Euclidean Distance Transform, y and y(hat) are reference and predicted
    segmentation maps and N is the total number of voxels in the volume.
    beta is a parameter that weights the two contriution accounting for the strong imbalance between the number of
    contour voxels and background voxels, defined as follows:
    .. math::
        \\beta = \\frac{N_{edge}}{N_{total}}

    where N_edge is the number of contour voxels.

    Args:
        numClasses: number of classes
    Returns:
        dist_region_crossentropy_loss function.
    """

    def dist_region_crossentropy_loss(y_true, y_pred):
        """
        Computes distanced "double-faced" regional cross entropy weighted with Euclidean Distance Map,
        after generation of Euclidean Distance Map with function "calc_DM_batch".
        Args:
            y_true: ground truth tensor of dimensions [class, batch_size, rows, columns]
            y_pred: A tensor resulting from a softmax of the same shape as y_true
        Returns:
            value of distanced "double-faced" regional cross entropy
        """

        if len(y_true.shape) == 5:
            axisSum = (1, 2, 3)
            y_pred = tf.transpose(y_pred, [4, 0, 1, 2, 3])
            y_true = tf.transpose(y_true, [4, 0, 1, 2, 3])
        elif len(y_true.shape) == 4:
            axisSum = (1, 2)
            y_pred = tf.transpose(y_pred, [3, 0, 1, 2])
            y_true = tf.transpose(y_true, [3, 0, 1, 2])
        else:
            print("Could not recognise input dimensions")
            return

        # Now dimensions are --> (numClasses, batchSize, Rows, Columns, Slices)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nVoxels = tf.size(y_true) / numClasses
        nVoxels = tf.cast(nVoxels, tf.float32)

        SDM = tf.py_function(func=calc_DM_batch,
                             inp=[y_true, numClasses],
                             Tout=tf.float32)
        gamma = 8
        sigma = 10
        DWM = 1 + gamma * tf.math.exp(tf.math.negative(SDM) / sigma)

        loss_weights = get_loss_weights(y_true, nVoxels, numClasses)
        epsilon = backend_config.epsilon
        epsilon_ = constant_op.constant(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        first_term = 0.5 * tf.multiply(DWM, tf.multiply(y_true, math_ops.log(y_pred)))
        second_term = 0.5 * tf.multiply(DWM, tf.multiply((1 - y_true), math_ops.log(1 - y_pred)))

        bc_temp = - tf.reduce_sum(tf.add(first_term, second_term), axis=(1, 2, 3, 4))
        bc = tf.reduce_sum(tf.multiply(loss_weights, bc_temp)) / nVoxels

        return bc

    return dist_region_crossentropy_loss





