from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
class test():
    def __init__(self):
        # Configure data loader
        self.dataset_name = 'black_to_blonde'
        self.data_loader = DataLoader(dataset_name=self.dataset_name)
    
    def sample_images(self,iter):
        os.makedirs('images/%s_test' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_test_img(domain ="A",iter=iter)
        imgs_B = self.data_loader.load_test_img(domain="B", iter=iter)
        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        print('Saving image')
        fig.savefig("images/%s_test/%s.png" % (self.dataset_name,iter))
        plt.close()

    def test(self,size,g_AB_path,g_BA_path):
        print('We are loading model')
        # Configure model used to test
        self.g_AB = load_model(g_AB_path,custom_objects={'InstanceNormalization':InstanceNormalization})
        self.g_BA = load_model(g_BA_path,custom_objects={'InstanceNormalization':InstanceNormalization})
        print('Model loading done')
        
        # perform perdiction
        for i in range(size):
            self.sample_images(i)

if __name__ == '__main__':
    g_AB_path = sys.argv[1]
    g_BA_path = sys.argv[2]
    size = int(sys.argv[3])

    test = test()
    test.test(size,g_AB_path,g_BA_path)
