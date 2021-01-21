"""
The cnn submodule implements some classes to define CNN-based
models in a dynamic way.
"""
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Model

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Conv3DTranspose
from keras.layers.pooling import AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D, MaxPool3D
from keras.layers import Input, Concatenate, Lambda, Dropout, Concatenate, Multiply, Softmax, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from segmentation.utils import conv_factory, transition, denseblock, channelModule, \
    spatialModule, denseUnit, compressionUnit, upsamplingUnit, squeeze_excite_block, \
    conv_block, encoder1, encoder2, decoder1, decoder2, output_block, Upsample, ASPP, PEE, RA, MINI_MTL, CFF, build_MINI_MTL
from tensorflow.keras.applications import *
import math


class UNet(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters. 
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size = (128,128,128,1), 
                        kernel_size = (3,3,3),
                        strides = (1,1,1),
                        deconv_kernel_size = (2,2,2),
                        deconv_strides = (2,2,2),
                        pool_size = (2,2,2),
                        pool_strides = (2,2,2),
                        depth = 5,
                        activation = 'relu',
                        padding = 'same',
                        n_initial_filters = 8,
                        add_batch_normalization = True,
                        kernel_regularizer = regularizers.l2(0.001),
                        bias_regularizer = regularizers.l2(0.001),
                        n_classes = 3):

        self.input_size = input_size
        self.n_dim = len(input_size) # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1,1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1,1,1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer
        
        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters*pow(2,i), kernel_size = self.kernel_size, 
                                            strides = self.strides, 
                                            padding = self.padding,
                                            activation = 'linear', 
                                            kernel_regularizer = self.kernel_regularizer, 
                                            bias_regularizer = self.bias_regularizer)(temp_layer)
                # batch normalization
                if (self.add_batch_normalization):
                    temp_layer = layers.BatchNormalization(axis = -1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size = self.pool_size,
                                         strides=self.pool_strides, 
                                         padding=self.padding)(temp_layer)
        for j in range(2): 
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters*pow(2,self.depth), kernel_size = self.kernel_size, 
                                        strides = self.strides, 
                                        padding = self.padding,
                                        activation = 'linear',
                                        kernel_regularizer = self.kernel_regularizer,
                                        bias_regularizer = self.bias_regularizer)(temp_layer)
            if (self.add_batch_normalization):
                temp_layer = temp_layer = layers.BatchNormalization(axis = -1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
        
        # Up sampling branch
        for i in range(self.depth):
            temp_layer = conv_transpose_layer(self.n_initial_filters*pow(2,(self.depth-1)-i),
                                                    kernel_size=self.deconv_kernel_size,
                                                    strides=self.deconv_strides,
                                                    activation='linear',
                                                    padding=self.padding,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
            # concatenation
            temp_layer = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth-1)-i], temp_layer])
            # convolution
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters*pow(2,(self.depth-1)-i),
                                            kernel_size = self.kernel_size, 
                                            strides = self.strides, 
                                            padding = self.padding,
                                            activation = 'linear', 
                                            kernel_regularizer = self.kernel_regularizer, 
                                            bias_regularizer = self.bias_regularizer)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
  

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size = softmax_kernel_size,
                                                       strides = self.strides,
                                                       padding = 'same', 
                                                       activation = 'linear',
                                                       kernel_regularizer = self.kernel_regularizer, 
                                                       bias_regularizer = self.bias_regularizer)(temp_layer)

        output_tensor = layers.Softmax(axis = -1)(temp_layer)
        self.model = Model(inputs = [input_tensor], outputs = [output_tensor])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise
    
    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class UNet2(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        input_layer = conv_layer(self.n_initial_filters, kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
        # batch normalization
        if (self.add_batch_normalization):
            input_layer = layers.BatchNormalization(axis=-1)(input_layer)
        # activation
        input_layer = layers.Activation(self.activation)(input_layer)

        temp_layer = max_pool_layer(pool_size=self.pool_size,
                                    strides=self.pool_strides,
                                    padding=self.padding)(input_layer)

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if (self.add_batch_normalization):
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)
        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            if (self.add_batch_normalization):
                temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_merge = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if (self.add_batch_normalization):
                    temp_layer_edge = layers.BatchNormalization(axis=-1)(temp_layer_edge)
                temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if (self.add_batch_normalization):
                    temp_layer_mask = layers.BatchNormalization(axis=-1)(temp_layer_mask)
                temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            if i % 2 != 1:
                # temp_layer_1 = RA(temp_layer_edge, temp_layer_mask, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                # temp_layer_2 = RA(temp_layer_mask, temp_layer_edge, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                # temp_layer_merge = Add()([temp_layer_1, temp_layer_2])
                # temp_layer_edge = PEE(temp_layer_edge, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                temp_layer_merge = Add()([temp_layer_edge, temp_layer_mask])


                # temp_layer_merge = Concatenate()([temp_layer_edge, temp_layer_mask])
                # temp_layer_merge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                #                               kernel_size=self.kernel_size,
                #                               strides=self.strides,
                #                               padding=self.padding,
                #                               activation='linear',
                #                               kernel_regularizer=self.kernel_regularizer,
                #                               bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            else:
                temp_layer_merge = temp_layer_mask

        ######################## ++++++++++++++++++++++++++++ #############################

        temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                               kernel_size=self.deconv_kernel_size,
                                               strides=self.deconv_strides,
                                               activation='linear',
                                               padding=self.padding,
                                               kernel_regularizer=self.kernel_regularizer,
                                               bias_regularizer=self.bias_regularizer)(temp_layer_edge)
        temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

        # MASK PATH
        temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                               kernel_size=self.deconv_kernel_size,
                                               strides=self.deconv_strides,
                                               activation='linear',
                                               padding=self.padding,
                                               kernel_regularizer=self.kernel_regularizer,
                                               bias_regularizer=self.bias_regularizer)(temp_layer_merge)
        temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

        temp_layer_edge = layers.Concatenate(axis=self.n_dim)([input_layer, temp_layer_edge])
        temp_layer_mask = layers.Concatenate(axis=self.n_dim)([input_layer, temp_layer_mask])

        temp_layer_edge = conv_layer(self.n_initial_filters,
                                     kernel_size=self.kernel_size,
                                     strides=self.strides,
                                     padding=self.padding,
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)
        if (self.add_batch_normalization):
            temp_layer_edge = layers.BatchNormalization(axis=-1)(temp_layer_edge)
        temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

        temp_layer_mask = conv_layer(self.n_initial_filters,
                                     kernel_size=self.kernel_size,
                                     strides=self.strides,
                                     padding=self.padding,
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_mask)
        if (self.add_batch_normalization):
            temp_layer_mask = layers.BatchNormalization(axis=-1)(temp_layer_mask)
        temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

        ######################## ++++++++++++++++++++++++++++ #############################

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_mask)

        temp_layer_edge = conv_layer(self.n_classes - 1, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        output_tensor_edge = layers.Softmax(axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = layers.Softmax(axis=-1, dtype='float32',  name='out_mask')(temp_layer_mask)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor_edge, output_tensor_mask])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class PEENet2(object):
    """
    This class provides a simple interface to create
    a U-Net network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)
        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_merge = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)([downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if self.add_batch_normalization:
                    temp_layer_edge = layers.BatchNormalization(axis=-1)(temp_layer_edge)
                temp_layer_edge = layers.Activation(self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if self.add_batch_normalization:
                    temp_layer_mask = layers.BatchNormalization(axis=-1)(temp_layer_mask)
                temp_layer_mask = layers.Activation(self.activation)(temp_layer_mask)

            if i % 2 != 1:
                # temp_layer_1 = RA(temp_layer_edge, temp_layer_mask, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                # temp_layer_2 = RA(temp_layer_mask, temp_layer_edge, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                # temp_layer_merge = Add()([temp_layer_1, temp_layer_2])
                temp_layer_edge = PEE(temp_layer_edge, self.n_initial_filters * pow(2, (self.depth - 1) - i))
                # temp_layer_merge = Add()([temp_layer_edge, temp_layer_mask])

                temp_layer_merge = Concatenate()([temp_layer_edge, temp_layer_mask])
                temp_layer_merge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                              kernel_size=self.kernel_size,
                                              strides=self.strides,
                                              padding=self.padding,
                                              activation='linear',
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            else:
                temp_layer_merge = temp_layer_mask

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        temp_layer_edge = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        output_tensor_edge = layers.Softmax(axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = layers.Softmax(axis=-1, dtype='float32',  name='out_mask')(temp_layer_mask)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor_edge, output_tensor_mask])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class ERANet(object):
    """
    This class provides a simple interface to create
    a ERANet (Edge Reverse Attention) network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch
        for i in range(self.depth):
            for j in range(2 + math.floor(i/2)):
                # Convolution
                if j == 0:
                    x = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=(1,1,1),
                                   strides=self.strides,
                                   padding=self.padding,
                                   activation='linear',
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer)(temp_layer)
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if (self.add_batch_normalization):
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            temp_layer = layers.Add()([temp_layer, x])
            downsampling_layers.append(temp_layer)
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)

        # Bottleneck
        temp_layer = ASPP(temp_layer, self.n_initial_filters * pow(2, self.depth))

        # Up sampling branch
        for i in range(self.depth):
            temp_layer = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                              kernel_size=self.deconv_kernel_size,
                                              strides=self.deconv_strides,
                                              activation='linear',
                                              padding=self.padding,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

            out_pee = PEE(downsampling_layers[(self.depth - 1) - i], self.n_initial_filters * pow(2, (self.depth - 1) - i))
            temp_layer = RA(temp_layer, out_pee, self.n_initial_filters * pow(2, (self.depth - 1) - i))

            # convolution
            for j in range(2 + math.floor(i/2)):
                # Convolution
                if j == 0:
                    x = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                   kernel_size=(1, 1, 1),
                                   strides=self.strides,
                                   padding=self.padding,
                                   activation='linear',
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer)(temp_layer)
                temp_layer = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)
            # addition
            temp_layer = layers.Add()([temp_layer, x])

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                strides=self.strides,
                                padding='same',
                                activation='linear',
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer)(temp_layer)

        output_tensor = layers.Softmax(axis=-1)(temp_layer)
        self.model = Model(inputs=[input_tensor], outputs=[output_tensor])

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class BAUNet(object):
    """
    This class provides a simple interface to create
    a BAUNet (Boundary Aware U) network with custom parameters.
    Args:
        input_size: input size for the network
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(128, 128, 128, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size)
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        out_edge_list = []
        out_mask_list = []
        out_mtl_list = []
        upsampling_layers = []

        # Down sampling branch
        for i in range(self.depth):
            for j in range(2):
                # Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i), kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # batch normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
                # activation
                temp_layer = layers.Activation(self.activation)(temp_layer)

            if i > 0:
                out_pee = PEE(temp_layer, self.n_initial_filters * pow(2, i))
                out_mtl, out_edge, out_mask = MINI_MTL(out_pee,
                                                       self.n_initial_filters * pow(2, i),
                                                       self.n_classes,
                                                       i)
                out_edge_list.append(out_edge)
                out_mask_list.append(out_mask)
                out_mtl_list.append(out_mtl)
            else:
                downsampling_layers.append(temp_layer)

            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)

        # Bottleneck
        temp_layer = ASPP(temp_layer, self.n_initial_filters * pow(2, self.depth))

        # Up sampling branch
        for i in range(self.depth):
            temp_layer = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                              kernel_size=self.deconv_kernel_size,
                                              strides=self.deconv_strides,
                                              activation='linear',
                                              padding=self.padding,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bias_regularizer=self.bias_regularizer)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)
            # IF MINI_MTL is used
            if i != (self.depth - 1):  # don't do that in the shallowest layer
                out_cff = CFF(out_mtl_list,
                              self.input_size[1],
                              self.n_initial_filters * pow(2, (self.depth - 1) - i),
                              (self.depth - 1) - i)

                # Concatenation
                temp_layer = Concatenate()([temp_layer, out_cff])
            else:  # simple concatenation in the shallowest layer
                temp_layer = Concatenate()([temp_layer, downsampling_layers[(self.depth - i) - 1]])

            temp_layer = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                    kernel_size=(1, 1, 1),
                                    padding='same')(temp_layer)

            # # Convolution
            # for j in range(2):
            #     # Convolution
            #     temp_layer = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
            #                             kernel_size=self.kernel_size,
            #                             strides=self.strides,
            #                             padding=self.padding,
            #                             activation='linear',
            #                             kernel_regularizer=self.kernel_regularizer,
            #                             bias_regularizer=self.bias_regularizer)(temp_layer)
            #     if self.add_batch_normalization:
            #         temp_layer = layers.BatchNormalization(axis=-1)(temp_layer)
            #     # activation
            #     temp_layer = layers.Activation(self.activation)(temp_layer)


        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                strides=self.strides,
                                padding='same',
                                activation='linear',
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer)(temp_layer)

        # out_edge = Concatenate()(out_edge_list)
        # out_edge = conv_layer(self.n_classes - 1, kernel_size=softmax_kernel_size,
        #                       strides=self.strides,
        #                       padding='same',
        #                       activation='linear',
        #                       kernel_regularizer=self.kernel_regularizer,
        #                       bias_regularizer=self.bias_regularizer)(out_edge)
        #
        # out_mask = Concatenate()(out_mask_list)
        # out_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
        #                       strides=self.strides,
        #                       padding='same',
        #                       activation='linear',
        #                       kernel_regularizer=self.kernel_regularizer,
        #                       bias_regularizer=self.bias_regularizer)(out_mask)

        # out_edge = layers.Softmax(axis=-1, name='out_edge')(out_edge)
        # out_mask = layers.Softmax(axis=-1, name='out_mask')(out_mask)
        output_tensor = layers.Softmax(axis=-1, dtype='float32', name='out_final')(temp_layer)

        self.model = Model(inputs=[input_tensor], outputs=[output_tensor] + out_edge_list + out_mask_list)

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class VNet(object):
    """
    This class provides a simple interface to create a 
    VNet network with custom parameters.
    """
    def __init__(self, input_size=(128,128,128)):
        pass


class CDDUnet(object):
    """
    This class provides a simple interface to create
    a Contextual Deconvolutional Dense Net network with custom parameters.
    Args:
        input_size: input size for the network
        num_classes: number of classes in labels
        weight_decay: weight decay parameter
        growth_rate: number of feature maps produced by each layer in the dense blocks
        n_layers: number of layers in the first dense block. Layers in the subsequent blocks are defined
            according to this parameter.
        theta: fraction to reduce number of features in the transition blocks after each dense block.
        activation: activation function used in the U-Net layers
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
    """

    def __init__(self, input_size,
                 num_classes,
                 weight_decay=1E-4,
                 growth_rate=12,
                 n_layers=6,
                 theta=0.5,
                 activation = 'relu',
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 ):

        self.input_size = input_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.theta = theta
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.n_filters_0 = self.growth_rate/2
        self.n_filters_1 = self.growth_rate
        self.n_filters_2 = self.growth_rate * 2


    def create_model(self):

        input_layer = layers.Input(shape=self.input_size)

        # Convolutional layer 0
        x = Conv3D(self.n_filters_0, self.kernel_size, padding='same')(input_layer)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_0 = Activation(self.activation, name='encoding_0')(x)  # 192x192x192x6

        # MaxPooling + Conv layer 1
        x = MaxPool3D(self.pool_size)(encoding_0)  # 96x96x96x6
        x = Conv3D(self.n_filters_1, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_1 = Activation(self.activation, name='encoding_1')(x)  # 96x96x96x12

        # MaxPooling + Conv layer 2
        x = MaxPool3D(self.pool_size)(encoding_1)  # 48x48x48x12
        x = Conv3D(self.n_filters_2, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_2 = Activation(self.activation, name='encoding_2')(x)  # 48x48x48x24

        # Dense Block 3
        x, n_block_3 = denseblock(encoding_2, concat_axis=-1, nb_layers=self.n_layers, nb_filter=self.n_filters_2,
                                  growth_rate=self.growth_rate)  # 48x48x48x96
        encoding_3, n_filters_3 = transition(x, concat_axis=-1, nb_filter=n_block_3,
                                             theta=self.theta)  # 24x24x24x48 (theta=0.5)

        # Dense Block 4
        x, n_block_4 = denseblock(encoding_3, concat_axis=-1, nb_layers=2 * self.n_layers, nb_filter=n_filters_3,
                                  growth_rate=self.growth_rate)  # 24x24x24x192
        encoding_4, n_filters_4 = transition(x, concat_axis=-1, nb_filter=n_block_4,
                                             theta=self.theta)  # 12x12x12x96 (theta=0.5)

        # Dense Block 5
        x, n_block_5 = denseblock(encoding_4, concat_axis=-1, nb_layers=4 * self.n_layers, nb_filter=n_filters_4,
                                  growth_rate=self.growth_rate)  # 12x12x12x348
        encoding_5, n_filters_5 = transition(x, concat_axis=-1, nb_filter=n_block_5, theta=self.theta)  # 6x6x6x192

        decoding_5 = compressionUnit(encoding_5, n_filters_4, kernel_size=self.kernel_size)  # 6x6x6x96

        # First concatenation
        x = upsamplingUnit(encoding_4, decoding_5, n_filters_4, n_filters_4, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_4 = compressionUnit(x, n_filters_3, kernel_size=self.kernel_size)  # 12x12x12x48

        # Second concatenation
        x = upsamplingUnit(encoding_3, decoding_4, n_filters_3, n_filters_3, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 24x24x24x96
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_3 = compressionUnit(x, self.n_filters_2, kernel_size=self.kernel_size)  # 24x24x24x24

        # Third concatenation
        x = upsamplingUnit(encoding_2, decoding_3, self.n_filters_2, self.n_filters_2, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 48x48x48x48
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_2 = compressionUnit(x, self.n_filters_1)  # 48x48x48x12

        # Fourth concatenation
        x = upsamplingUnit(encoding_1, decoding_2, self.n_filters_1, self.n_filters_1, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 96x96x96x24
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_1 = compressionUnit(x, self.n_filters_0)  # 96x96x96x6

        # Last concatenation
        x = upsamplingUnit(encoding_0, decoding_1, self.n_filters_0, self.n_filters_0, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 192x192x192x12

        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.num_classes, (1, 1, 1), padding='same', kernel_initializer='he_uniform')(x)
        output_layer = Softmax(axis=-1)(x)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])

    def set_initial_weights(self, weights):
        """
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        """
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        """
        Get the total number of parameters of the model
        """
        return self.model.count_params()

    def summary(self):
        """
        Print out summary of the model.
        """
        print(self.model.summary())


class CDUnet(object):
    """
    This class provides a simple interface to create
    a Contextual Deconvolutional UNet network with custom parameters.
    Same as CDDUnet only without dense connections in dowsampling path
    to reduce memory requirements.
    Args:
        input_size: input size for the network
        num_classes: number of classes in labels
        weight_decay: weight decay parameter
        initial_filters: number of initial filters that defines subsequent number of features for convolutions
        activation: activation function used in the U-Net layers
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
    """

    def __init__(self, input_size,
                 num_classes,
                 weight_decay=1E-4,
                 initial_filters = 6,
                 activation = 'relu',
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 ):

        self.input_size = input_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.initial_filters = initial_filters
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.n_filters_0 = self.initial_filters
        self.n_filters_1 = self.initial_filters * 2
        self.n_filters_2 = self.initial_filters * 4
        self.n_filters_3 = self.initial_filters * 8
        self.n_filters_4 = self.initial_filters * 16
        self.n_filters_5 = self.initial_filters * 32


    def create_model(self):

        input_layer = layers.Input(shape=self.input_size)

        # Conv layer 0
        x = Conv3D(self.n_filters_0, self.kernel_size, padding='same')(input_layer)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_0, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_0 = Activation(self.activation, name='encoding_0')(x)  # 192x192x192x6

        # MaxPooling_0
        max_pool_0 = MaxPool3D(self.pool_size)(encoding_0)  # 96x96x96x6

        # Conv layer 1
        x = Conv3D(self.n_filters_1, self.kernel_size, padding='same')(max_pool_0)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_1, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_1 = Activation(self.activation, name='encoding_1')(x)  # 96x96x96x12

        # Maxpooling_1
        max_pool_1 = MaxPool3D(self.pool_size)(encoding_1)  # 48x48x48x12

        # Conv layer 2
        x = Conv3D(self.n_filters_2, self.kernel_size, padding='same')(max_pool_1)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_2, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_2 = Activation(self.activation, name='encoding_2')(x)  # 48x48x48x24

        # Maxpooling_2
        max_pool_2 = MaxPool3D(self.pool_size)(encoding_2)  # 24x24x24x24

        # Conv layer 3
        x = Conv3D(self.n_filters_3, self.kernel_size, padding='same')(max_pool_2)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_3, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_3 = Activation(self.activation, name='encoding_3')(x)  # 24x24x24x48

        # Maxpooling_3
        max_pool_3 = MaxPool3D(self.pool_size)(encoding_3)  # 12x12x12x48

        # Conv layer 3
        x = Conv3D(self.n_filters_4, self.kernel_size, padding='same')(max_pool_3)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_4, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_4 = Activation(self.activation, name='encoding_4')(x)  # 12x12x12x96

        # Maxpooling_4
        max_pool_4 = MaxPool3D(self.pool_size)(encoding_4)  # 6x6x6x96

        # Conv layer 4
        x = Conv3D(self.n_filters_5, self.kernel_size, padding='same')(max_pool_4)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.n_filters_5, self.kernel_size, padding='same')(x)
        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        encoding_5 = Activation(self.activation, name='encoding_5')(x)  # 6x6x6x192  # BOTTLENECK

        decoding_5 = compressionUnit(encoding_5, self.n_filters_4, kernel_size=self.kernel_size)  # 6x6x6x96

        # First concatenation
        x = upsamplingUnit(encoding_4, decoding_5, self.n_filters_4, self.n_filters_4, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_4 = compressionUnit(x, self.n_filters_3, kernel_size=self.kernel_size)

        # Second concatenation
        x = upsamplingUnit(encoding_3, decoding_4, self.n_filters_3, self.n_filters_3, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 24x24x24x96
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_3 = compressionUnit(x, self.n_filters_2, kernel_size=self.kernel_size)  # 24x24x24x24

        # Third concatenation
        x = upsamplingUnit(encoding_2, decoding_3, self.n_filters_2, self.n_filters_2, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 48x48x48x48
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_2 = compressionUnit(x, self.n_filters_1, kernel_size=self.kernel_size)  # 48x48x48x12

        # Fourth concatenation
        x = upsamplingUnit(encoding_1, decoding_2, self.n_filters_1, self.n_filters_1, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 96x96x96x24
        x = denseUnit(x, kernel_size=self.kernel_size)
        x = denseUnit(x, kernel_size=self.kernel_size)
        decoding_1 = compressionUnit(x, self.n_filters_1, kernel_size=self.kernel_size)  # 96x96x96x6

        # Last concatenation
        x = upsamplingUnit(encoding_0, decoding_1, self.n_filters_0, self.n_filters_0, kernel_size=self.kernel_size,
                           deconv_kernel_size=self.deconv_kernel_size, deconv_strides=self.deconv_strides)  # 192x192x192x12

        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation(self.activation)(x)

        x = Conv3D(self.num_classes, (1, 1, 1), padding='same', kernel_initializer='he_uniform')(x)
        output_layer = Softmax(axis=-1)(x)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])

    def set_initial_weights(self, weights):
        """
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        """
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        """
        Get the total number of parameters of the model
        """
        return self.model.count_params()

    def summary(self):
        """
        Print out summary of the model.
        """
        print(self.model.summary())


def createCDDUnet(input_shape, NumClasses, weight_decay=1E-4, growth_rate=12, n_layers=6, theta=0.5):
    """
    Function to create a contextual deconvolution dense segmentation model with spatial and
        channels modules in the decoding path. 
    Args:
        input_shape: shape of input tensor
        NumClasses: number of classes to segment
        weight_decay: weight decay parameter
        growth_rate: number of feature maps produced by each layer in the dense blocks
        n_layers: number of layers in the first dense block. Layers in the subsequent blocks are defined 
            according to this parameter.
        theta: fraction to reduce number of features in the transition blocks after each dense block. 

    Returns: softmax probability maps of segmentation masks for the numClasses number of classes specified in input.
    """

    input_layer = layers.Input(shape=input_shape)

    n_filters_0 = growth_rate/2
    n_filters_1 = growth_rate
    n_filters_2 = 2 * growth_rate

    # Convolutional layer 0
    x = Conv3D(n_filters_0, (3, 3, 3), padding='same')(input_layer)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_0 = Activation('relu', name='encoding_0')(x)  # 192x192x192x6

    # MaxPooling + Conv layer 1
    x = MaxPool3D((2, 2, 2))(encoding_0)  # 96x96x96x6
    x = Conv3D(n_filters_1, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_1 = Activation('relu', name='encoding_1')(x)  # 96x96x96x12

    # MaxPooling + Conv layer 2
    x = MaxPool3D((2, 2, 2))(encoding_1)  # 48x48x48x12
    x = Conv3D(n_filters_2, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_2 = Activation('relu', name='encoding_2')(x)  # 48x48x48x24

    # Dense Block 3
    x, n_block_3 = denseblock(encoding_2, concat_axis=-1, nb_layers=n_layers, nb_filter=n_filters_2,
                                growth_rate=growth_rate)  # 48x48x48x96
    encoding_3, n_filters_3 = transition(x, concat_axis=-1, nb_filter=n_block_3, theta=theta)  # 24x24x24x48 (theta=0.5)


    # Dense Block 4
    x, n_block_4 = denseblock(encoding_3, concat_axis=-1, nb_layers=2*n_layers, nb_filter=n_filters_3,
                                growth_rate=growth_rate)  # 24x24x24x192
    encoding_4, n_filters_4 = transition(x, concat_axis=-1, nb_filter=n_block_4, theta=theta)  # 12x12x12x96 (theta=0.5)

    # Dense Block 5
    x, n_block_5 = denseblock(encoding_4, concat_axis=-1, nb_layers=4*n_layers, nb_filter=n_filters_4,
                                growth_rate=growth_rate)  # 12x12x12x348
    encoding_5, n_filters_5 = transition(x, concat_axis=-1, nb_filter=n_block_5, theta=theta)  # 6x6x6x192

    decoding_5 = compressionUnit(encoding_5, n_filters_4)  # 6x6x6x96

    # First concatenation
    x = upsamplingUnit(encoding_4, decoding_5, n_filters_4, n_filters_4)
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_4 = compressionUnit(x, n_filters_3)  # 12x12x12x48

    # Second concatenation
    x = upsamplingUnit(encoding_3, decoding_4, n_filters_3, n_filters_3)  # 24x24x24x96
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_3 = compressionUnit(x, n_filters_2)  # 24x24x24x24

    # Third concatenation
    x = upsamplingUnit(encoding_2, decoding_3, n_filters_2, n_filters_2)  # 48x48x48x48
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_2 = compressionUnit(x, n_filters_1)  # 48x48x48x12

    # Fourth concatenation
    x = upsamplingUnit(encoding_1, decoding_2, n_filters_1, n_filters_1)  # 96x96x96x24
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_1 = compressionUnit(x, n_filters_1)  # 96x96x96x6

    # Last concatenatiokn
    x = upsamplingUnit(encoding_0, decoding_1, n_filters_0, n_filters_0)  # 192x192x192x12

    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(NumClasses, (1, 1, 1), padding='same', kernel_initializer='he_uniform')(x)
    output_layer = Softmax(axis=-1)(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


def createCDUnet(input_shape, NumClasses, weight_decay=1E-4, initial_filters=6, n_layers=6, theta=0.5):
    """
    Function to create a contextual deconvolution segmentation model with spatial and
        channels modules in the decoding path.
    Args:
        input_shape: shape of input tensor
        NumClasses: number of classes to segment
        weight_decay: weight decay parameter
        initial_filters: number of initial filters that defines subsequent number of features for convolutions
        n_layers: number of layers in the first dense block. Layers in the subsequent blocks are defined
            according to this parameter.
        theta: fraction to reduce number of features in the transition blocks after each dense block.

    Returns: softmax probability maps of segmentation masks for the numClasses number of classes specified in input.
    """
    input_layer = layers.Input(shape=input_shape)

    n_filters_0 = initial_filters
    n_filters_1 = 2 * initial_filters
    n_filters_2 = 4 * initial_filters
    n_filters_3 = 8 * initial_filters
    n_filters_4 = 16 * initial_filters
    n_filters_5 = 32 * initial_filters

    # Conv layer 0
    x = Conv3D(n_filters_0, (3, 3, 3), padding='same')(input_layer)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_0, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_0 = Activation('relu', name='encoding_0')(x)  # 192x192x192x6

    # MaxPooling_0
    max_pool_0 = MaxPool3D((2, 2, 2))(encoding_0)  # 96x96x96x6

    # Conv layer 1
    x = Conv3D(n_filters_1, (3, 3, 3), padding='same')(max_pool_0)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_1, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_1 = Activation('relu', name='encoding_1')(x)  # 96x96x96x12

    # Maxpooling_1
    max_pool_1 = MaxPool3D((2, 2, 2))(encoding_1)  # 48x48x48x12

    # Conv layer 2
    x = Conv3D(n_filters_2, (3, 3, 3), padding='same')(max_pool_1)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_2, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_2 = Activation('relu', name='encoding_2')(x)  # 48x48x48x24

    # Maxpooling_2
    max_pool_2 = MaxPool3D((2, 2, 2))(encoding_2)  # 24x24x24x24

    # Conv layer 3
    x = Conv3D(n_filters_3, (3, 3, 3), padding='same')(max_pool_2)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_3, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_3 = Activation('relu', name='encoding_3')(x)  # 24x24x24x48

    # Maxpooling_3
    max_pool_3 = MaxPool3D((2, 2, 2))(encoding_3)  # 12x12x12x48

    # Conv layer 3
    x = Conv3D(n_filters_4, (3, 3, 3), padding='same')(max_pool_3)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_4, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_4 = Activation('relu', name='encoding_4')(x)  # 12x12x12x96

    # Maxpooling_4
    max_pool_4 = MaxPool3D((2, 2, 2))(encoding_4)  # 6x6x6x96

    # Conv layer 4
    x = Conv3D(n_filters_5, (3, 3, 3), padding='same')(max_pool_4)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters_5, (3, 3, 3), padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    encoding_5 = Activation('relu', name='encoding_5')(x)  # 6x6x6x192  # BOTTLENECK

    decoding_5 = compressionUnit(encoding_5, n_filters_4)  # 6x6x6x96

    # First concatenation
    x = upsamplingUnit(encoding_4, decoding_5, n_filters_4, n_filters_4)
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_4 = compressionUnit(x, n_filters_3)

    # Second concatenation
    x = upsamplingUnit(encoding_3, decoding_4, n_filters_3, n_filters_3)  # 24x24x24x96
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_3 = compressionUnit(x, n_filters_2)  # 24x24x24x24

    # Third concatenation
    x = upsamplingUnit(encoding_2, decoding_3, n_filters_2, n_filters_2)  # 48x48x48x48
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_2 = compressionUnit(x, n_filters_1)  # 48x48x48x12

    # Fourth concatenation
    x = upsamplingUnit(encoding_1, decoding_2, n_filters_1, n_filters_1)  # 96x96x96x24
    x = denseUnit(x)
    x = denseUnit(x)
    decoding_1 = compressionUnit(x, n_filters_1)  # 96x96x96x6

    # Last concatenation
    x = upsamplingUnit(encoding_0, decoding_1, n_filters_0, n_filters_0)  # 192x192x192x12

    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)

    x = Conv3D(NumClasses, (1, 1, 1), padding='same', kernel_initializer='he_uniform')(x)
    output_layer = Softmax(axis=-1)(x)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


def build_doubleUnet(shape, numClasses):
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 48)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = inputs * outputs1

    x, skip_2 = encoder2(x)
    x = ASPP(x, 48)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])

    outputs = Conv3D(numClasses, (1, 1, 1), padding="same")(outputs)
    outputs = Softmax(axis = -1)(outputs)

    model = Model(inputs, outputs)
    return model




