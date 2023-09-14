# ---------------------------------------------------------------------
# Disclaimer: IMPORTANT: This software was developed at the National 
# Institute of Standards and Technology by employees of the Federal 
# Government in the course of their official duties. Pursuant to
# title 17 Section 105 of the United States Code this software is 
# not subject to copyright protection and is in the public domain. 
# This is an experimental system. NIST assumes no responsibility 
# whatsoever for its use by other parties, and makes no guarantees, 
# expressed or implied, about its quality, reliability, or any other 
# characteristic. We would appreciate acknowledgement if the software 
# is used. This software can be redistributed and/or modified freely 
# provided that any derivative works bear some notice that they are 
# derived from it, and any modified versions bear some notice that 
# they have been modified.
# ---------------------------------------------------------------------
import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    print('Tensorflow 2.x.x required')
    sys.exit(1)

#from utils import Deconv3D, Conv3D, BN_ReLU, Dilated_Conv3D, multihead_attention_3d

class UNet():
    _BASELINE_FEATURE_DEPTH = 64
    _KERNEL_SIZE = 3
    _DECONV_KERNEL_SIZE = 2
    _POOLING_STRIDE = 2

    SIZE_FACTOR = 16
    RADIUS = 96  # nearest multiple of 16 over 92 pixels radius required from the unet paper ((572 - 388) / 2 = 92)

    # 08/30/19: make changes below to run with channels last to run on CPU

    @staticmethod
    def _conv_layer3d(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv3D(filters=filter_count,
                                        kernel_size=kernel,
                                        strides=stride,
                                        padding='same',
                                        activation=tf.keras.activations.relu,  # 'relu'
                                        data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output
    @staticmethod
    def _conv_layer(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv2D(filters=filter_count,
                                        kernel_size=kernel,
                                        strides=stride,
                                        padding='same',
                                        activation=tf.keras.activations.relu,  # 'relu'
                                        data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _deconv_layer3d(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv3DTranspose(filters=filter_count,
                                                 kernel_size=kernel,
                                                 strides=stride,
                                                 activation=None,
                                                 padding='same',
                                                 data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _deconv_layer(input, filter_count, kernel, stride=1):
        output = tf.keras.layers.Conv2DTranspose(filters=filter_count,
                                                 kernel_size=kernel,
                                                 strides=stride,
                                                 activation=None,
                                                 padding='same',
                                                 data_format='channels_first')(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def _pool3d(input, size):
        pool = tf.keras.layers.MaxPool3D(pool_size=size, data_format='channels_first')(input)
        return pool
    @staticmethod
    def _pool(input, size):
        pool = tf.keras.layers.MaxPool2D(pool_size=size, data_format='channels_first')(input)
        return pool

    @staticmethod
    def _concat(input1, input2, axis):
        output = tf.keras.layers.Concatenate(axis=axis)([input1, input2])
        return output

    @staticmethod
    def _dropout(input):
        output = tf.keras.layers.Dropout(rate=0.5)(input)
        return output

    def __init__(self, number_classes, global_batch_size, img_size, class_weights, learning_rate=3e-4, label_smoothing=0):

        self.img_size = img_size
        self.learning_rate = learning_rate
        self.number_classes = number_classes
        self.global_batch_size = global_batch_size

        self.class_weights = class_weights
        self.weights = tf.constant(self.class_weights)
        print('tfweights',self.weights)
        # create a tf tensor with the weights
        #self.weightstf = self.get_weights_tensor()

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        print('imgsize',img_size)
        img_size = [1,img_size[0],img_size[1],img_size[2]]
        #self.inputs = tf.keras.Input(shape=(img_size[0], None, None, None))
        self.inputs = tf.keras.Input(shape=(1,16,128,128))
        self.model = self._build_model()

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model2(self):
        # Encoder
        print('image shape---',self.inputs.shape)
        conv_1 = UNet._conv_layer3d(self.inputs, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        return conv_1

    def _build_model(self):
        # Encoder
        print('image shape---',self.inputs.shape)
        conv_1 = UNet._conv_layer3d(self.inputs, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_1 = UNet._conv_layer3d(conv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('conv1',conv_1.shape)
        pool_1 = UNet._pool3d(conv_1, UNet._POOLING_STRIDE)

        conv_2 = UNet._conv_layer3d(pool_1, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_2 = UNet._conv_layer3d(conv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('conv2',conv_2.shape)

        pool_2 = UNet._pool3d(conv_2, UNet._POOLING_STRIDE)

        conv_3 = UNet._conv_layer3d(pool_2, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_3 = UNet._conv_layer3d(conv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('conv3',conv_3.shape)

        pool_3 = UNet._pool3d(conv_3, UNet._POOLING_STRIDE)

        conv_4 = UNet._conv_layer3d(pool_3, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_4 = UNet._conv_layer3d(conv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('conv4',conv_4.shape)
        conv_4 = UNet._dropout(conv_4)

        pool_4 = UNet._pool3d(conv_4, UNet._POOLING_STRIDE)

        # bottleneck
        bottleneck = UNet._conv_layer3d(pool_4, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        bottleneck = UNet._conv_layer3d(bottleneck, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('bottle',bottleneck.shape)
        bottleneck = UNet._dropout(bottleneck)

        # Decoder
        # up-conv which reduces the number of feature channels by 2
        deconv_4 = UNet._deconv_layer3d(bottleneck, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_4 = UNet._concat(conv_4, deconv_4, axis=1)
        #deconv_4 = UNet._concat(conv_4, deconv_4, axis=4) # for HWC version
        deconv_4 = UNet._conv_layer3d(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_4 = UNet._conv_layer3d(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('dconv4',deconv_4.shape)

        deconv_3 = UNet._deconv_layer3d(deconv_4, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_3 = UNet._concat(conv_3, deconv_3, axis=1)
        #deconv_3 = UNet._concat(conv_3, deconv_3, axis=4)
        deconv_3 = UNet._conv_layer3d(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_3 = UNet._conv_layer3d(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('dconv3',deconv_3.shape)

        deconv_2 = UNet._deconv_layer3d(deconv_3, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_2 = UNet._concat(conv_2, deconv_2, axis=1)
        #deconv_2 = UNet._concat(conv_2, deconv_2, axis=4)
        deconv_2 = UNet._conv_layer3d(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_2 = UNet._conv_layer3d(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('dconv2',deconv_2.shape)

        deconv_1 = UNet._deconv_layer3d(deconv_2, UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        deconv_1 = UNet._concat(conv_1, deconv_1, axis=1)
        #deconv_1 = UNet._concat(conv_1, deconv_1, axis=4)
        deconv_1 = UNet._conv_layer3d(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_1 = UNet._conv_layer3d(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        print('dconv1',deconv_1.shape)

        logits = UNet._conv_layer3d(deconv_1, self.number_classes, 1)  # 1x1 kernel to convert feature map into class map
        # convert NCHW to NHWC so that softmax axis is the last dimension
        print('logits',logits.shape)
        # already NWHC for CPU inferencing: don't need this
        logits = tf.keras.layers.Permute((2, 3, 4, 1))(logits)
        # logits is [NHWC]

        softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)
        print('softmax',softmax.shape)
        unet = tf.keras.Model(self.inputs, softmax, name='unet')

        return unet

    def _build_model_old(self):

        # Encoder
        print('image shape---',self.inputs.shape)
        conv_1 = UNet._conv_layer(self.inputs, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_1 = UNet._conv_layer(conv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_1 = UNet._pool(conv_1, UNet._POOLING_STRIDE)

        conv_2 = UNet._conv_layer(pool_1, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_2 = UNet._conv_layer(conv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_2 = UNet._pool(conv_2, UNet._POOLING_STRIDE)

        conv_3 = UNet._conv_layer(pool_2, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_3 = UNet._conv_layer(conv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        pool_3 = UNet._pool(conv_3, UNet._POOLING_STRIDE)

        conv_4 = UNet._conv_layer(pool_3, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_4 = UNet._conv_layer(conv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        conv_4 = UNet._dropout(conv_4)

        pool_4 = UNet._pool(conv_4, UNet._POOLING_STRIDE)

        # bottleneck
        bottleneck = UNet._conv_layer(pool_4, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        bottleneck = UNet._conv_layer(bottleneck, 16 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        bottleneck = UNet._dropout(bottleneck)

        # Decoder
        # up-conv which reduces the number of feature channels by 2
        deconv_4 = UNet._deconv_layer(bottleneck, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        #deconv_4 = UNet._concat(conv_4, deconv_4, axis=1)
        deconv_4 = UNet._concat(conv_4, deconv_4, axis=3)
        deconv_4 = UNet._conv_layer(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_4 = UNet._conv_layer(deconv_4, 8 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_3 = UNet._deconv_layer(deconv_4, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        #deconv_3 = UNet._concat(conv_3, deconv_3, axis=1)
        deconv_3 = UNet._concat(conv_3, deconv_3, axis=3)
        deconv_3 = UNet._conv_layer(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_3 = UNet._conv_layer(deconv_3, 4 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_2 = UNet._deconv_layer(deconv_3, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        #deconv_2 = UNet._concat(conv_2, deconv_2, axis=1)
        deconv_2 = UNet._concat(conv_2, deconv_2, axis=3)
        deconv_2 = UNet._conv_layer(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_2 = UNet._conv_layer(deconv_2, 2 * UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        deconv_1 = UNet._deconv_layer(deconv_2, UNet._BASELINE_FEATURE_DEPTH, UNet._DECONV_KERNEL_SIZE, stride=UNet._POOLING_STRIDE)
        #deconv_1 = UNet._concat(conv_1, deconv_1, axis=1)
        deconv_1 = UNet._concat(conv_1, deconv_1, axis=3)
        deconv_1 = UNet._conv_layer(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)
        deconv_1 = UNet._conv_layer(deconv_1, UNet._BASELINE_FEATURE_DEPTH, UNet._KERNEL_SIZE)

        logits = UNet._conv_layer(deconv_1, self.number_classes, 1)  # 1x1 kernel to convert feature map into class map
        # convert NCHW to NHWC so that softmax axis is the last dimension
        print('logits',logits.shape)
        logits = tf.keras.layers.Permute((2, 3, 1))(logits)
        # logits is [NHWC]

        softmax = tf.keras.layers.Softmax(axis=-1, name='softmax')(logits)
        print('softmax',softmax.shape)
        unet = tf.keras.Model(self.inputs, softmax, name='unet')

        return unet

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, labels, wts, loss_metric, accuracy_metric) = inputs

        loss_value = tf.constant(0.0)
        with tf.GradientTape() as tape:
            softmax = self.model(images, training=True)
            
            loss_value = self.loss_fn(labels, softmax) # [NxHxWx1]
            loss_value = tf.multiply(loss_value,wts)

            # average across the batch (N) with the approprite global batch size
            loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size
            

            # reduce down to a scalar (reduce H, W)
            loss_value = tf.reduce_mean(loss_value)
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)
        accuracy_metric.update_state(labels, softmax)

        return loss_value


    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        self.inputs = inputs
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, labels, wts, loss_metric, accuracy_metric) = inputs
        softmax = self.model(images, training=False)

        loss_value = self.loss_fn(labels, softmax)
        loss_value = tf.multiply(loss_value,wts)

        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size
        # reduce down to a scalar (reduce H, W)
        loss_value = tf.reduce_mean(loss_value)

        loss_metric.update_state(loss_value)
        accuracy_metric.update_state(labels, softmax)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value
