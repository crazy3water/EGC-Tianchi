import keras.layers as layers
import keras.backend as backend
import keras.models as models
from keras import backend as K
import My_layer

def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 20, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 2
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv1D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = My_layer.MaxBlurPooling1D(pool_size=1,kernel_size=5,name=name)(x)
    x = Spatial_pyramid_pooling(x, name=name + '_pool')
    # x = layers.AveragePooling1D(2, strides= 2, name=name + '_pool')(x)
    # x = layers.MaxPooling1D(2, strides=2, name=name + '_pool')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    bn_axis = 2

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv1D(2 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    # x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
    #                                name=name + '_1_bn')(x1)
    # x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = identity_block(x1, 3, 2 * growth_rate, name, 'block')
    # x1 = layers.Conv1D(growth_rate, 3, #(1,3)
    #                    padding='same',
    #                    use_bias=False,
    #                    name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def Spatial_pyramid_pooling(x,name):

    x_max_2 = layers.MaxPooling1D(2, strides=2, name=name + '_SPP_max2',padding='same')(x)
    _, W, C = K.int_shape(x_max_2)

    x_max_3 = layers.MaxPooling1D(4, strides=4, name=name + '_stride_4',padding='same')(x)
    _, W1, _ = K.int_shape(x_max_3)
    x_max_3_r = layers.ZeroPadding1D(padding=(0, W - W1) )(x_max_3)

    x_max_4 = layers.MaxPooling1D(8, strides=8, name=name + '_stride_8',padding='same')(x)
    _, W1, _ = K.int_shape(x_max_4)
    x_max_4_r = layers.ZeroPadding1D(padding=(0, W - W1) )(x_max_4)

    x = layers.Concatenate(name=name+ '3_4')([x_max_3_r,x_max_4_r])
    x = layers.Concatenate(name=name + '_all')([x_max_2,x])

    x = layers.Conv1D(
        filters=C,
        kernel_size=1,
        strides=1,
        padding="same",
    )(x)

    return x

def self_Att_channel(x,x_att,r = 16,name = '1'):
    '''
    advanced
    Hu, Jie, Li Shen, and Gang Sun."Squeeze-and-excitation networks." arXiv preprintarXiv:1709.01507 (2017).
    :param x:
    :param r:
    :return:
    '''
    x_self = x
    chanel = K.int_shape(x)[-1]
    L = K.int_shape(x)[-2]

    x_att = layers.GlobalAveragePooling1D(name='self_avg_pool' + name )(x_att)

    # x_att = layers.Conv2D(chanel,
    #                       (H,W),
    #                       padding='valid',
    #                       use_bias=None,
    #                       name='FCN' + name)(x_att)

    x_att = layers.Dense(int(chanel / r),activation='relu')(x_att)
    x_att = layers.Dense(chanel, activation='sigmoid')(x_att)
    x = layers.Multiply()([x_self,x_att])

    return x



def DenseNet(blocks,
             include_top=True,
             input_shape=None,
             pooling=None,
             classes=1000
             ):

    # Determine proper input shape
    bn_axis = 2

    img_input = layers.Input(shape=input_shape)

    # x = layers.Reshape((input_shape[0], input_shape[2],input_shape[1]))(img_input)
    # x = layers.Reshape((input_shape[1], input_shape[0]))(img_input)


    x = img_input

    # x = Melspectrogram(n_dft=128, n_hop=64, input_shape=(input_shape[0], input_shape[1]),
    #                          padding='same', sr=500, n_mels=80,
    #                          fmin=40.0, fmax=500/2, power_melgram=1.0,
    #                          return_decibel_melgram=True, trainable_fb=False,
    #                          trainable_kernel=False,
    #                          name='mel_stft') (x)
    #
    # x = layers.BatchNormalization(
    #     axis=bn_axis, epsilon=1.001e-5, name= 'spectrogram/bn')(x)
    # x = layers.Permute((2, 1, 3))(x)

    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # x = layers.Conv2D(64, (1,3), strides=(1,1), use_bias=False, name='conv1/conv1',padding='same')(x)

    # x = layers.BatchNormalization(
    #     axis=bn_axis, epsilon=1.001e-5, name='conv0/bn')(x)

    x = layers.Conv1D(64, 10, strides=3, use_bias=False, name='conv1/conv2', padding='same')(x)
    # x = layers.Conv2D(64, (1, 3), strides=(1, 2), use_bias=False, name='conv1/conv', padding='valid')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    x = self_Att_channel(x,x_att= x, r=4, name='1')

    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # x = layers.MaxPooling2D((1,2), strides=(1,2), name='pool1')(x1)

    x1 = dense_block(x, blocks[0], name='conv2')
    x1 = transition_block(x1, 0.5, name='pool2')
    x = self_Att_channel(x1, x_att=x1, r=4, name='2')

    x2 = dense_block(x, blocks[1], name='conv3')
    x2 = transition_block(x2, 0.5, name='pool3')
    x = self_Att_channel(x2, x_att=x2, r=4, name='3')

    x3 = dense_block(x, blocks[2], name='conv4')
    x3 = transition_block(x3, 0.5, name='pool4')
    x = self_Att_channel(x3, x_att=x3, r=4, name='4')

    x4 = dense_block(x, blocks[3], name='conv5')
    x4 = transition_block(x4, 0.5, name='pool5')
    x = self_Att_channel(x4, x_att=x4, r=4, name='5')

    if include_top:

        # x = layers.Reshape([1,W,chanel*H],name = 'final_reshape')(x)
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)

        x = layers.Dense(classes, activation='sigmoid', name=str(classes))(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors
    #
    #  of `input_tensor`.
    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    return model
