# from __future__ import print_function, division

# import tensorflow as tf
# from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add, Concatenate, Subtract
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers.core import Lambda
from keras.models import Sequential, Model
from keras.applications.vgg19 import VGG19
# from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop, Adam
from keras import regularizers
#from sklearn.cross_validation import train_test_split  # 
from functools import partial
import keras.backend as K
import matplotlib.pyplot as plt

# from data_process import dataProcess, PreProcess

import sys
import os
import numpy as np
import datetime
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True #
# session = tf.Session(config=config)
# KTF.set_session(session) #

batch_size = 64
sep = "/"
full_model_name = os.path.basename(__file__)
(model_name, suffix) = os.path.splitext(full_model_name)
BASE_DIR = 'weights_' + model_name + '/'


def check_path(path):
    if not os.path.lexists(path):
        os.mkdir(path)


check_path(BASE_DIR)

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return(weights*inputs[0]) + ((1-weights)*inputs[1])


class ImprovedWGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.low_img_shape = (self.img_rows, self.img_cols, self.channels)
        self.real_shape = (self.img_rows, self.img_cols, 1)
        self.real_shape3 = (self.img_rows, self.img_cols, 3)
        self.n_critic = 5
        self.filters= 64

        d_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        g_opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

        self.generator = self.build_generator()
        self.discriminator = self.build_discrimintor()
        self.generator.load_weights('weights_wgan_filter64//117//wgan_filter64_generator_305_0.h5')
        self.discriminator.load_weights('weights_wgan_filter64//117//wgan_filter64_discriminator_305.h5')

        # 
        self.generator.trainable = False
        self.discriminator.trainable = True
        real_img = Input(shape=self.real_shape) 
        z_disc = Input(shape=self.low_img_shape) 
        fake_img = self.generator(z_disc) 

        fake = self.discriminator(fake_img)
        real = self.discriminator(real_img)
        merged_img = RandomWeightedAverage()([real_img, fake_img])
        valid_merged = self.discriminator(merged_img)
        partial_gp_loss = partial(self.gradient_penatly_loss,
                                  averaged_samples=merged_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = Model(inputs=[real_img, z_disc],
                                         outputs=[real, fake, valid_merged])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss],
                                         optimizer=d_opt,
                                         loss_weights=[1, 1, 10])
        # 
        self.discriminator.trainable = False
        self.generator.trainable = True
        
        z_gen = Input(shape=self.low_img_shape) #64,64,1
        img = self.generator(z_gen) #64,64,3
        valid = self.discriminator(img)
        self.generator_model = Model(z_gen,[valid, img, img, img])
        self.generator_model.compile(loss=[self.wasserstein_loss, 'mae', self.perceptual_loss, self.ssim_loss],
                                     optimizer=g_opt, loss_weights=[0.005, 1.8, 1, 1])
        


    def build_generator(self):
        inputs = Input(self.low_img_shape) 
        outputs = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv')(inputs)  
        outputs0 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv0')(outputs) 
        outputs1 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv1')(outputs0) 
        outputs2 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv2')(outputs1) 
        outputs3 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv3')(outputs2)  
        outputs4 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv4')(outputs3) 
        outputs5 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv5')(outputs4)  
        outputs6 = Conv2D(self.filters, 3, strides=(1, 1), activation='relu', padding='valid', name='conv6')(outputs5)  
        
        outputs6_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv6')(outputs6) 
        outputs6_r = Add()([outputs6_r, outputs5]) 
        outputs6_r = Activation('relu')(outputs6_r)  

        outputs5_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv5')(outputs6_r) 
        outputs5_r = Add()([outputs5_r, outputs4])  
        outputs5_r = Activation('relu')(outputs5_r) 

        outputs4_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv4')(outputs5_r)  
        outputs4_r = Activation('relu')(outputs4_r)  

        outputs3_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv3')(outputs4_r)  
        outputs3_r = Add()([outputs3_r, outputs2])  
        outputs3_r = Activation('relu')(outputs3_r)  

        outputs2_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv2')(outputs3_r) 
        outputs2_r = Activation('relu')(outputs2_r) 

        outputs1_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv1')(outputs2_r)  
        outputs1_r = Add()([outputs1_r, outputs0]) 
        outputs1_r = Activation('relu')(outputs1_r) 

        outputs0_r = Conv2DTranspose(self.filters, 3, strides=(1, 1), padding='valid', name='deconv0')(outputs1_r)  
        outputs0_r = Activation('relu')(outputs0_r) 

        outputs = Conv2DTranspose(1, 3, strides=(1, 1), padding='valid', name='deconv')(outputs0_r) 

        outputs = Add()([outputs, inputs]) 
        outputs = Activation('relu')(outputs)

        model = Model(inputs=inputs, outputs=outputs) 
        model.summary() 
        return model  



    def build_discrimintor(self):
        inputs = Input(self.low_img_shape)
        conv1 = Conv2D(64, 3, padding='same',
                       kernel_initializer='he_uniform')(inputs)
        conv1 = LeakyReLU()(conv1)
        conv2 = ZeroPadding2D(padding=(1, 1))(conv1)
        conv2 = Conv2D(64, 3, padding='valid', strides=2,
                       kernel_initializer='he_uniform')(conv2)
        conv2 = LeakyReLU()(conv2)
        conv3 = Conv2D(128, 3, padding='same',
                       kernel_initializer='he_uniform')(conv2)
        conv3 = LeakyReLU()(conv3)
        conv4 = ZeroPadding2D(padding=(1, 1))(conv3)
        conv4 = Conv2D(128, 3, padding='valid', strides=2,
                       kernel_initializer='he_uniform')(conv4)
        conv4 = LeakyReLU()(conv4)
        conv5 = Conv2D(256, 3, padding='same',
                       kernel_initializer='he_uniform')(conv4)
        conv5 = LeakyReLU()(conv5)
        conv6 = ZeroPadding2D(padding=(1, 1))(conv5)
        conv6 = Conv2D(256, 3, padding='valid', strides=2,
                       kernel_initializer='he_uniform')(conv6)
        conv6 = LeakyReLU()(conv6)
        flat6 = Flatten()(conv6)
        fc1 = Dense(1024)(flat6)
        fc1 = LeakyReLU()(fc1)
        fc2 = Dense(1)(fc1)
        model = Model(inputs = inputs, outputs=fc2)
        model.summary()
        return model
 


    def gradient_penatly_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_spr_sum = K.sum(gradients_sqr,
                                  axis = np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_spr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def perceptual_loss(self, y_true, y_pred):
        
        y_true = K.concatenate([y_true, y_true, y_true],axis=-1)
        y_pred = K.concatenate([y_pred, y_pred, y_pred],axis=-1)
        # print("******",y_true.get_shape())
        # print("*****",y_pred.get_shape())
        vgg = VGG19(include_top=False, weights='imagenet',
                    input_shape=self.real_shape3)
        loss_model = Model(inputs = vgg.input,
                           outputs=vgg.get_layer('block4_conv4').output)
        loss_model.trainable = False
        return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


    def save_all_weights(self, d, g, epoch_number, current_loss):
        now = datetime.datetime.now()
        save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        g.save_weights(os.path.join(save_dir, model_name + '_generator_{}_{}.h5'.format(
            epoch_number, current_loss)), True)
        d.save_weights(os.path.join(
            save_dir, model_name + '_discriminator_{}.h5'.format(epoch_number)), True)

    #
    # def load_data(self):
    #     mydata = dataProcess()
    #     (train_img, train_lbl) = mydata.load_train_data()
    #     print(train_img.shape, train_lbl.shape)
    #     return train_img, train_lbl

    def load_data(self):
        img = np.load('train_data_crop_aug.npy')
        label = np.load('train_label_crop_aug.npy')
        return img[..., np.newaxis], label[..., np.newaxis]


    def train(self, epochs, batch_size, sample_interval=50):
        img_train, img_mask_train = self.load_data()
        z, w, h, c = img_train.shape

        print("train data: ", img_train.shape, img_train.mean(),
              img_train.min(), img_train.max())

        valid = -np.ones((batch_size, 1)) # for real img
        fake = np.ones((batch_size, 1))  #for fake img
        dummy = np.zeros((batch_size, 1)) #Dummy for gradient penalty
        for index in range(epochs): #
            print('epoch: {}/{}'.format(index, epochs))
            print('batches: {}'.format(img_train.shape[0]/batch_size))

            d_losses = []
            g_losses = []
            perceptual_losses = []
            L1_losses = []
            w_losses = []
            ssim_losses = []

            for epoch in range(int(img_train.shape[0]/batch_size)):#batch number
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, img_train.shape[0], batch_size)
                    imgs = img_mask_train[idx]
                    noise = img_train[idx]
                    d_loss = self.discriminator_model.train_on_batch([imgs, noise],
                                                                     [valid, fake, dummy])
                    d_losses.append(d_loss[0])
                idx = np.random.randint(0, img_train.shape[0], batch_size)
                noise = img_train[idx]
                imgs = img_mask_train[idx]
                g_loss = self.generator_model.train_on_batch(noise, [valid, imgs, imgs, imgs])
                g_losses.append(g_loss[0])
                # w_losses.append(g_loss[1])
                L1_losses.append(g_loss[2])
                perceptual_losses.append(g_loss[3]) 
                ssim_losses.append(g_loss[4])
                
                

                print('batch {} g_loss : {}'.format(epoch, g_loss))
                print('batch {} d_loss : {}'.format(epoch, d_loss))

                if epoch % sample_interval == 0:
                    self.sample_images(index, epoch, img_train, img_mask_train)
            
            
            
            with open(model_name + '_log.txt', 'a') as f:
                f.write('{}    {}    {}    {}    {}    {}\n'.format(index,
                                                        np.mean(d_losses), np.mean(g_losses), np.mean(L1_losses), np.mean(perceptual_losses),
                                                        np.mean(ssim_losses)))

            self.save_all_weights(
                self.discriminator, self.generator, index, int(np.mean(g_losses)))

    def ssim_loss(self, y_true, y_pred):
        mu_true = K.mean(y_true, axis=-1)
        mu_pred = K.mean(y_pred, axis=-1)
        var_true = K.var(y_true, axis=3)
        var_pred = K.var(y_pred, axis=3)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        k1 = 0.01
        k2 = 0.03

        c1 = k1 ** 2
        c2 = k2 ** 2   

        # Get std dev
        covar_true_pred = K.mean(y_true * y_pred,
                                 axis=-1) - mu_true * mu_pred
        ssim = (2 * mu_true * mu_pred + c1) * (2 * covar_true_pred + c2)
        denom = (K.square(mu_true) + K.square(mu_pred) + c1) * \
                (var_pred + var_true + c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)


    def sample_images(self, index, epoch, test, label):
        save_dir = "images_" + model_name 
        check_path(save_dir) 
        r, c = 1, 4 
        idx = np.random.randint(0, test.shape[0], 1)
        low_image = test[idx] #1 64 64 1
        
        nor_image = label[idx] # 1 64 64 3
        normal_does = nor_image[:, :, :, 0].reshape(1, self.img_rows, self.img_cols, 1)
        # print("**********", normal_does.shape)
        gen_image = self.generator.predict(low_image) # 1 64 64 3
        predict = gen_image[0, :, :, 0].reshape(1, self.img_rows, self.img_cols, 1)
        # print("**********", predict.shape)
        
        images = np.concatenate([low_image, predict, normal_does, low_image-predict])
        # print("**********", images.shape)
        titles = ['low_image', 'gen_image', 'nor_image', 'dif_image']
        fig, axs = plt.subplots(r, c) 
        
        axes = axs.flatten()
        axes[0].imshow(images[0].reshape(self.img_rows, self.img_cols)) 
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        axes[1].imshow(images[1].reshape(self.img_rows, self.img_cols)) 
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        axes[2].imshow(images[2].reshape(self.img_rows, self.img_cols)) 
        axes[2].set_title(titles[2])
        axes[2].axis('off')
        axes[3].imshow(images[3].reshape(self.img_rows, self.img_cols))
        axes[3].set_title(titles[3])
        axes[3].axis('off') 
        plt.tight_layout() 
        fig.savefig(save_dir + sep + model_name + "_e%d_s%d_low%d_gen%d_nor%d.png" %
                    (index, epoch, int(low_image.mean()*1000), int(gen_image.mean()*1000), int(nor_image.mean()*1000)),
                    bbox_inches='tight')
        plt.close()


    


if __name__ == '__main__':
    wgan = ImprovedWGAN()
    wgan.train(epochs=500, batch_size=batch_size, sample_interval=150)
    






