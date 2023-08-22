# -*- coding: UTF-8 -*-
"""
    Name: Qingneng Li (Eng: Lional)
    Time: 2020/06/10
    Place: SIAT, Shenzhen
    Item: umap --> CT
"""
import os, datetime, shutil

import skimage

from utils import *
from models import *
from numpy import *
import tensorflow as tf
from models import CBAMBlock
tf.config.run_functions_eagerly(True)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from config import FLAGS
from matplotlib import pyplot as plt
import pandas as pd
"""=================================== Configure ======================================"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import os
import numpy as np
import time
import inspect
# tf.debugging.set_log_device_placement(True)

# VGG = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

"""====================================== Main ======================================"""
from tensorflow import keras
from tensorflow.keras import layers, Model




def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))



VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, allow_pickle=True, encoding='latin1').item()
        # print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [-1, 1]
        """

        start_time = time.time()
        # print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = self.relu_layer(self.conv1_1, "relu1_1")
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = self.relu_layer(self.conv1_2, "relu1_2")
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = self.relu_layer(self.conv2_1, "relu2_1")
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = self.relu_layer(self.conv2_2, "relu2_2")
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = self.relu_layer(self.conv3_1, "relu3_1")
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = self.relu_layer(self.conv3_2, "relu3_2")
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = self.relu_layer(self.conv3_3, "relu3_3")
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = self.relu_layer(self.conv3_4, "relu3_4")
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = self.relu_layer(self.conv4_1, "relu4_1")
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = self.relu_layer(self.conv4_2, "relu4_2")
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = self.relu_layer(self.conv4_3, "relu4_3")
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = self.relu_layer(self.conv4_4, "relu4_4")
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.relu5_1 = self.relu_layer(self.conv5_1, "relu5_1")
        self.conv5_2 = self.conv_layer(self.relu5_1, "conv5_2")
        self.relu5_2 = self.relu_layer(self.conv5_2, "relu5_2")
        self.conv5_3 = self.conv_layer(self.relu5_2, "conv5_3")
        self.relu5_3 = self.relu_layer(self.conv5_3, "relu5_3")
        self.conv5_4 = self.conv_layer(self.relu5_3, "conv5_4")
        self.relu5_4 = self.relu_layer(self.conv5_4, "relu5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # self.data_dict = None
        # print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def relu_layer(self, bottom, name):
        return tf.nn.relu(bottom, name=name)

    def conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            #            relu = tf.nn.relu(bias)

            return bias

    def fc_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
class WGANGP():
    def __init__(self,vgg_model_path='./weights/vgg19.npy'):
        # load dataset
        # self.train_dataset = read_and_decode_png("D:\\DLspace\\cycleGAN-NL\\datasets\\gz_data_8000_s\\TFRecord\\input.tfrecord", True)
        # self.train_dataset = load_train("E://mydata")
        self.train_dataset = read_and_decode("./train1.tfrecord")
        self.val_dataset = read_and_decode_test("./val1.tfrecord")
        self.train_iter = iter(self.train_dataset)
        self.test_dataset = read_and_decode_test("./train1.tfrecord")
        self.test_iter = iter(self.test_dataset)
        self.val_dataset_len = None
        # Build G and D model
        self.gen_model = Residual_Unet()
        self.dis_model = D_WGANGP()
        self.gen_tv = self.gen_model.trainable_variables
        self.dis_tv = self.dis_model.trainable_variables
      #  self.gen_model.load_weights('D:\DLspace\Keras\logs\model_29.h5', by_name=True)

        # Hyper-paramenters in train phrase
        self.step_per_epoch = FLAGS.num_data // FLAGS.batch_size
        self.train_steps = FLAGS.num_epoch * self.step_per_epoch
        self.n_critic = 5
        self.vgg = Vgg19(vgg_model_path)

        # Build each G and D optimizers
        gen_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [10000,40000], [FLAGS.gen_lr,0.1*FLAGS.gen_lr,0.01*FLAGS.gen_lr,])
        dis_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [10000,40000], [FLAGS.dis_lr,0.1*FLAGS.dis_lr,0.01*FLAGS.dis_lr,])

        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr_schedule, 0.5, 0.9)
        self.dis_optimizer = tf.keras.optimizers.Adam(dis_lr_schedule, 0.5, 0.9)

        # Loss Weights
        self.LAMBDA_dw, self.LAMBDA_dg = 1, 10
        self.LAMBDA_gw, self.LAMBDA_gm = 1e-3, 100
        self.LAMBDA_gp, self.LAMBDA_gg, self.LAMBDA_gs = 1e-4, 50, 5 #1e3,50,5

        # Tensorboard
        self.summary_writer = tf.summary.create_file_writer(FLAGS.logdir)

    """================================= Loss ==================================="""
    def dis_wast_loss(self, D_real, D_fake):
        D_real_loss = -1 * tf.reduce_mean(D_real)
        D_fake_loss = tf.reduce_mean(D_fake)
        D_loss = self.LAMBDA_dw *(D_real_loss + D_fake_loss)
        return D_loss, D_real_loss, D_fake_loss
    def dis_grad_loss(self, real_label, fake_image):
        alpha = tf.random.uniform((FLAGS.batch_size, 1, 1, 1), 0., 1.)
        sample = real_label + alpha * (fake_image - real_label)
        with tf.GradientTape() as gradtap:
            gradtap.watch(sample)
            valid = self.dis_model(sample, training=True)
        gradients = gradtap.gradient(valid, sample)
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, [1, 2, 3])
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = tf.square(1.0 - gradient_l2_norm)
        return self.LAMBDA_dg * tf.reduce_mean(gradient_penalty)
    # def dis_grad_loss(self, real_label, fake_image):
    #     alpha = tf.random.uniform((FLAGS.batch_size, 1, 1, 1), 0., 1.)
    #     sample = real_label + alpha * (fake_image - real_label)
    #     with tf.GradientTape() as gradtap:
    #         gradtap.watch(sample)
    #         valid = self.dis_model(sample, training=True)
    #
    #     gradients = gradtap.gradient(valid, [sample])
    #     gradients = tf.reshape(gradients,[gradients.shape[0],-1])
    #     gp = tf.norm(gradients,axis=1)
    #     gp = tf.reduce_mean((gp-1)**2)
    #     # gradients_sqr = tf.square(gradients)
    #     # gradients_sqr_sum = tf.reduce_sum(gradients_sqr, [1, 2, 3])
    #     # gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    #     # gradient_penalty = tf.square(1.0 - gradient_l2_norm)
    #     return self.LAMBDA_dg * gp

    def gen_gan_loss(self, D_fake):
        gan_loss = -1 * tf.reduce_mean(D_fake)
        return self.LAMBDA_gw * gan_loss

    def gen_mse_loss(self, real_label, fake_image):
        # pixel_loss = tf.reduce_sum([
        #     1.00*self.BCE(real_label, fake_image),
        #     0.02*self.PCP(real_label, fake_image)])
        # L1
        #mae_loss = tf.reduce_mean(tf.abs(real_label-fake_image))
        mse_loss = tf.reduce_mean(tf.square(real_label-fake_image))
        return self.LAMBDA_gm * mse_loss

    # def gen_content(self, real_label, fake_image):
    #     conv1 = layers.Conv2D(64, 3, 3, 'same', kernel_initializer=k_init)(real_label)
    #     conv1_ = layers.Conv2D(64, 3, 3, 'same', kernel_initializer=k_init)(fake_image)
    #     content_1 = tf.nn.l2_loss(conv1 - conv1_) / tf.cast(tf.size(conv1), dtype=tf.float32)
    #
    #     conv2 = layers.Conv2D(64, 1, 1, 'same', kernel_initializer=k_init)(real_label)
    #     conv2_ = layers.Conv2D(64, 1, 1, 'same', kernel_initializer=k_init)(fake_image)
    #     content_2 = tf.nn.l2_loss(conv2 - conv2_) / tf.cast(tf.size(conv2), dtype=tf.float32)
    #
    #     conv3 = layers.Conv2D(64, 5, 5, 'same', kernel_initializer=k_init)(real_label)
    #     conv3_ = layers.Conv2D(64, 5, 5, 'same', kernel_initializer=k_init)(fake_image)
    #     content_3 = tf.nn.l2_loss(conv3 - conv3_) / tf.cast(tf.size(conv3), dtype=tf.float32)
    #
    #     content_loss = 0.2*content_1 + 0.7*content_2 + 0.1*content_3
    #     return self.LAMBDA_gc * content_loss

    # def gen_gdl_loss(self,real_label, fake_image):
    #     real_dy, real_dx = tf.image.image_gradients(real_label)
    #     fake_dy, fake_dx = tf.image.image_gradients(fake_image)
    #     dif_dy = tf.abs(tf.abs(fake_dy) - tf.abs(real_dy))
    #     dif_dx = tf.abs(tf.abs(fake_dx) - tf.abs(real_dx))
    #     gdl_loss = tf.reduce_mean(dif_dy + dif_dx)
    #     return self.LAMBDA_gg * gdl_loss
    def Perceptual_loss(self,label, logits,batchsize):

        self.vgg.build(tf.concat([label, logits], axis=0))
        conten_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(self.vgg.relu3_3[batchsize:] - self.vgg.relu3_3[:batchsize]), axis=3))
        return self.LAMBDA_gp*conten_loss


    def gen_ssim_loss(self,real_label, fake_image):
        logit = (fake_image + 1.) / 2.
        label = (real_label + 1.) / 2.

        ssim = tf.image.ssim(logit, label, max_val=1.0)
        # in fact, for [0,1] value range, ssim must be >0
        ssim_positive = tf.math.maximum(0., ssim)
        ssim_loss = tf.reduce_mean(-tf.math.log(ssim_positive))
        return self.LAMBDA_gs * ssim_loss
    """================================= Train ==================================="""
    def train(self):
        Trainpsnr_all = []
        Trainssim_all = []
        Trainpl_all = []
        TrainG_ssim_loss_all = []
        TrainG_perceptual_loss_1_all = []
        TrainG_mse_loss_all = []
        TrainG_gan_loss_all = []
        TrainG_loss_all = []
        TrainD_loss_all = []
        Trainpsnr_epoch = []
        Trainssim_epoch = []
        Trainpl_epoch = []
        TrainG_ssim_loss_epoch = []
        TrainG_perceptual_loss_1_epoch = []
        TrainG_mse_loss_epoch = []
        TrainG_gan_loss_epoch = []
        TrainG_loss_epoch = []
        TrainD_loss_epoch = []
        epoch_i = 0
        start_time = datetime.datetime.now()
        pre_epoch  = None
        for batch_i, (input, label) in enumerate(self.train_dataset):
            for i in range(self.n_critic):
                with tf.GradientTape() as dis_tape:
                    d_input, d_label = next(self.train_iter)
                    d_logit = self.gen_model(d_input, training=True)
                    d_real = self.dis_model(d_label, training=True)
                    d_fake = self.dis_model(d_logit, training=True)
                    D_wast_loss = self.dis_wast_loss(d_real, d_fake)
                    D_grad_loss = self.dis_grad_loss(d_label, d_logit)
                    D_loss = D_wast_loss[0] + D_grad_loss
                # Compute gradients and apply gradients to optimizer.
                dis_gradients = dis_tape.gradient(D_loss, self.dis_tv)
                self.dis_optimizer.apply_gradients(zip(dis_gradients, self.dis_tv))

            with tf.GradientTape() as gen_tape:
                logit = self.gen_model(input, training=True)
                D_fake = self.dis_model(logit, training=True)
                G_gan_loss = self.gen_gan_loss(D_fake)
                G_mse_loss = self.gen_mse_loss(label, logit)
                newlogit = tf.concat([(logit+1.0)/2.0,(logit+1.0)/2.0,(logit+1.0)/2.0],axis=3)
                newlabel = tf.concat([(label+1.0)/2.0, (label+1.0)/2.0, (label+1.0)/2.0], axis=3)
                newinput = tf.concat([(input+1.0)/2.0, (input+1.0)/2.0, (input+1.0)/2.0], axis=3)
                G_perceptual_loss_1 = self.Perceptual_loss(newlabel, newlogit,newlogit.shape[0])
                G_perceptual_loss_1 = G_perceptual_loss_1.numpy()
                # G_gdl_loss = self.gen_gdl_loss(label,logit)
                G_ssim_loss = self.gen_ssim_loss(label, logit)
                #Loss all
                G_loss = G_gan_loss + G_perceptual_loss_1 + G_ssim_loss + G_mse_loss
                # G_loss = G_gan_loss + G_mse_loss + G_perceptual_loss_1

                # Compute Metrics: PSNR, SSIM, PCC in self.generator_g
                # nrmse = NRMSE((logit+1.0)/2.0, (label+1.0)/2.0)
                psnr = PSNR((logit+1.0)/2.0, (label+1.0)/2.0) # please change
                psnr = psnr.numpy()
                ssim = SSIM((logit+1.0)/2.0, (label+1.0)/2.0) # please change
                ssim = ssim.numpy()
                # pcc = PCC((logit+1.0)/2.0, (label+1.0)/2.0) # please change

            # Compute gradients and apply gradients to optimizer.
            gen_gradients = gen_tape.gradient(G_loss, self.gen_tv)
            self.gen_optimizer.apply_gradients(zip(gen_gradients, self.gen_tv))

            elapsed_time = datetime.datetime.now() - start_time
            print ("[Step %d/%d] [D loss: %.4g/%.4g] [G loss: %.4g/%.4g] "
                   "[PSNR/SSIM/PL: %.2f/%.4f/%.4g] time: %s"
                   % (batch_i, self.train_steps, D_wast_loss[1], D_wast_loss[2],
                      G_gan_loss, G_loss, psnr, ssim , G_perceptual_loss_1, elapsed_time))

            Trainpsnr_all.append(psnr)
            Trainssim_all.append(ssim)
            Trainpl_all.append(G_perceptual_loss_1)
            TrainG_ssim_loss_all.append(G_ssim_loss)
            TrainG_perceptual_loss_1_all.append(G_perceptual_loss_1)
            TrainG_mse_loss_all.append(G_mse_loss)
            TrainG_gan_loss_all.append(G_gan_loss)
            TrainG_loss_all.append(G_loss)
            TrainD_loss_all.append(D_loss)


            if (batch_i+1) % self.step_per_epoch ==0:
                psnr = mean(Trainpsnr_all[epoch_i * 500:])
                Trainpsnr_epoch.append(psnr)
                ssim = mean(Trainssim_all[epoch_i * 500:])
                Trainssim_epoch.append(ssim)
                pl = mean(Trainpl_all[epoch_i * 500:])
                Trainpl_epoch.append(pl)

                G_ssim_loss = mean(TrainG_ssim_loss_all[epoch_i * 500:])
                TrainG_ssim_loss_epoch.append(G_ssim_loss)
                G_perceptual_loss_1 = mean(TrainG_perceptual_loss_1_all[epoch_i * 500:])
                TrainG_perceptual_loss_1_epoch.append(G_perceptual_loss_1)
                G_mse_loss = mean(TrainG_mse_loss_all[epoch_i * 500:])
                TrainG_mse_loss_epoch.append(G_mse_loss)
                G_gan_loss = mean(TrainG_gan_loss_all[epoch_i * 500:])
                TrainG_gan_loss_epoch.append(G_gan_loss)
                G_loss = mean(TrainG_loss_all[epoch_i * 500:])
                TrainG_loss_epoch.append(G_loss)
                D_loss = mean(TrainD_loss_all[epoch_i * 500:])
                TrainD_loss_epoch.append(D_loss)

                epoch_i += 1
            epoch = (batch_i + 1) // self.step_per_epoch

            if pre_epoch!=epoch:
                pre_epoch = epoch
                # self.validate(epoch)


                with self.summary_writer.as_default():
                    # tf.summary.scalar('Learning_rate/G', FLAGS.gen_lr, epoch)
                    # tf.summary.scalar('Learning_rate/D', FLAGS.dis_lr, epoch)
                    # tf.summary.scalar('D_loss/Total', D_loss, epoch)
                    # tf.summary.scalar('D_loss/real', D_wast_loss[1], epoch)
                    # tf.summary.scalar('D_loss/fake', D_wast_loss[2], epoch)
                    # tf.summary.scalar('D_loss/grad', D_grad_loss, epoch)
                    tf.summary.scalar('G_loss/Total', G_loss, epoch)
                    # tf.summary.scalar('G_loss/gan', G_gan_loss, epoch)
                    # tf.summary.scalar('G_loss/mse', G_mse_loss, epoch)
                    tf.summary.scalar('Metric/PSNR', psnr, epoch)
                    tf.summary.scalar('Metric/SSIM', ssim, epoch)
                    # tf.summary.scalar('Metric/PL', G_perceptual_loss_1, epoch)
                    tf.summary.image('Input', (input+1.)/2.0, epoch, 4)
                    tf.summary.image('Label', (label+1.)/2.0, epoch, 4)
                    tf.summary.image('Logit', (logit+1.)/2.0, epoch, 4)
                    # save the model after training the generator reloaded in the future
                    if ((epoch+1) % 10== 0):
                        self.gen_model.save(FLAGS.logdir + '/model_%02d.h5'%(epoch+1))

            if (batch_i+1) == self.train_steps:
                self.gen_model.save(FLAGS.logdir + '/model.h5')
                break
        # return TrainG_perceptual_loss_1_epoch,TrainG_ssim_loss_epoch,TrainG_mse_loss_epoch,TrainD_loss_epoch
    def validate(self,epoch):

        ValG_loss_all = []
        Val_PSNR_all = []
        Val_ssim_all = []


        start_time = datetime.datetime.now()
        if self.val_dataset_len is None:
            self.val_dataset_len = 0
            for example in self.val_dataset:
                self.val_dataset_len += 1
        # generator.summary()
        for batch_i, (input, label) in enumerate(self.val_dataset):
            logit = self.gen_model(input, training=False)
            D_fake = self.dis_model(logit, training=False)
            G_gan_loss = self.gen_gan_loss(D_fake)
            G_mse_loss = self.gen_mse_loss(label, logit)
            newlogit = tf.concat([(logit + 1.0) / 2.0, (logit + 1.0) / 2.0, (logit + 1.0) / 2.0], axis=3)
            newlabel = tf.concat([(label + 1.0) / 2.0, (label + 1.0) / 2.0, (label + 1.0) / 2.0], axis=3)
            G_perceptual_loss_1 = self.Perceptual_loss(newlabel, newlogit, newlogit.shape[0])
            G_perceptual_loss_1 = G_perceptual_loss_1.numpy()
            # G_gdl_loss = self.gen_gdl_loss(label,logit)
            G_ssim_loss = self.gen_ssim_loss(label, logit)
            # Loss all
            G_loss = G_gan_loss + G_perceptual_loss_1 + G_ssim_loss + G_mse_loss
            psnr = PSNR((logit + 1.0) / 2.0, (label + 1.0) / 2.0)  # please change
            psnr = psnr.numpy()
            ssim = SSIM((logit + 1.0) / 2.0, (label + 1.0) / 2.0)  # please change
            ssim = ssim.numpy()

            elapsed_time = datetime.datetime.now() - start_time
            print("[val][Step %d/%d] [G loss: %.4g/%.4g] "
                  "[PSNR/SSIM/PL: %.2f/%.4f/%.4g] time: %s"
                  % (batch_i, self.val_dataset_len,
                     G_gan_loss, G_loss, psnr, ssim, G_perceptual_loss_1, elapsed_time))

            ValG_loss_all.append(G_loss)
            Val_PSNR_all.append(psnr)
            Val_ssim_all.append(ssim)




        print("--------------验证集G_LOSS平均值为： ", mean(ValG_loss_all))
        print("--------------验证集PSNR平均值为： ", mean(Val_PSNR_all))
        print("--------------验证集SSIM平均值为： ", mean(Val_ssim_all))


        with self.summary_writer.as_default():
            tf.summary.scalar('Metric/G_loss_val', mean(ValG_loss_all), epoch)
            tf.summary.scalar('Metric/PSNR_val', mean(Val_PSNR_all), epoch)
            tf.summary.scalar('Metric/SSIM_val', mean(Val_ssim_all), epoch)






    def test(self):
        start_time = time.time()
        Testpsnr_all = []
        Testssim_all = []
        Testpl_all = []
        NMI_all = []
        generator = tf.keras.models.load_model(FLAGS.logdir + '/model_10.h5', custom_objects={'CBAMBlock': CBAMBlock,
                                                                                               'CBAMBlock.downsample':lambda x:x})  # 加载模型
        # generator.summary()
        for batch_i, (input, label) in enumerate(self.test_dataset):

            logit = generator.predict(input)
            logit = tf.clip_by_value(logit, -1, 1)
            for i in range(len(logit)):
                result_NMI = skimage.metrics.normalized_mutual_information(logit[i], label[i], bins=100)
                print(f"result_NMI{i}:", result_NMI)
                NMI_all.append(result_NMI)
                ind = batch_i * len(logit) + i
                tf.keras.preprocessing.image.save_img(
                    'test_img/test_a11' + '/'  + '_%05d.jpg' % ind,
                    tf.concat([input[i], logit[i], label[i]], 1))

                # Compute Metrics: PSNR, SSIM, PCC in self.generator_g
            # nrmse = NRMSE((logit + 1.0) / 2.0, (label + 1.0) / 2.0)
            # nmae = NMAE((logit + 1.0) / 2.0, (label + 1.0) / 2.0)
            psnr = PSNR((logit + 1.0) / 2.0, (label + 1.0) / 2.0)  # please change
            psnr = psnr.numpy()
            ssim = SSIM((logit + 1.0) / 2.0, (label + 1.0) / 2.0)  # please change
            ssim = ssim.numpy()
            newlogit = tf.concat([(logit + 1.0) / 2.0, (logit + 1.0) / 2.0, (logit + 1.0) / 2.0], axis=3)
            newlabel = tf.concat([(label + 1.0) / 2.0, (label + 1.0) / 2.0, (label + 1.0) / 2.0], axis=3)
            G_perceptual_loss = self.Perceptual_loss(newlabel, newlogit, newlogit.shape[0])
            G_perceptual_loss = G_perceptual_loss.numpy()

            # print("[PSNR/SSIM/PL: %.2f/%.4f/%.4g] "
            #       % (psnr, ssim, G_perceptual_loss))

            Testpsnr_all.append(psnr)
            Testssim_all.append(ssim)
            Testpl_all.append(G_perceptual_loss)


        # print(np.mean(Testpsnr_all),np.mean(Testssim_all),np.mean(Testpl_all))
        # print(np.std(Testpsnr_all), np.std(Testssim_all), np.std(Testpl_all))
        print(np.mean(NMI_all),"-------",np.std(NMI_all))

        # list = np.array([Testpsnr_all, Testssim_all, Testpl_all]).T
        list = np.array([NMI_all]).T
        # name = ["psnr", "ssim", "pl"]
        name = ["nmi"]
        df = pd.DataFrame(columns=name, data=list)
        df.to_csv(r"C:\Users\Colonel\Desktop\实验记录\test\nmi5.csv")

    def test_attention(self):
        generator = tf.keras.models.load_model(FLAGS.logdir + '/model_100.h5', custom_objects={'CBAMBlock': CBAMBlock,
                                                                                               'CBAMBlock.downsample': lambda
                                                                                                   x: x})  # 加载模型
        input = cv2.imread("pic/0input.jpg",cv2.IMREAD_GRAYSCALE)/255 #np数组
        logit = generator.predict(input.reshape(1,512,512,1))
        logit = tf.clip_by_value(logit, -1, 1)
        logit = np.array(logit)
        print(logit.shape)

        cv2.imwrite("test/attention/a0.jpg",((logit.reshape(512,512)+1)/2*255).astype(np.uint8))
        bb = cv2.imread("test/attention/a0.jpg",cv2.IMREAD_GRAYSCALE)
        aa = cv2.cvtColor(bb ,cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test/attention/a1.jpg",aa)


if __name__ == '__main__':
    # if os.path.exists(FLAGS.logdir):
    #     shutil.rmtree(FLAGS.logdir)           #递归删除一个目录以及目录内的所有内容
    gan = WGANGP()
    # gan.train()
    gan.test()
    # gan.test_attention()