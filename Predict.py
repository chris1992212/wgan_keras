import os
import numpy as np
from time import time
import pydicom
from keras.layers import Input, Dense, Activation

from keras.layers.merge import Add, Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from natsort import ns, natsorted

# from keras.models import load_model
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def check_path(path):
    """
    check path if it is exist. otherwise create this path.
    """
    if not os.path.lexists(path):
        os.makedirs(path)


def get_dicom_data(path):
    low_list = []
    file_list = os.listdir(path)
    for file in file_list:
        if os.path.splitext(file)[-1] == '.dcm':
            ds_low = pydicom.dcmread(os.path.join(path, file))
            low_mat = ds_low.pixel_array
            low_list.append(low_mat)
    low_data = np.array(low_list)
    low_data = np.expand_dims(low_data, axis=-1)
    low_data = ((low_data + 1000) * 1 / 4000).clip(0, 1)
    return low_data


class TestPredict(object):
    def __init__(self,
                 predict_img="./predict/data/10",
                 width=512,
                 height=512,
                 output_name="predict_result"):

        self.predict_img = predict_img
        self.width = width
        self.height = height
        self.output_name = output_name
        self.low_img_shape = self.width, self.height, 1
        check_path(output_name)

    def build_generator(self):
        self.filters = 64
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

    def predict_img_data(self):
        t0 = time()
        low_image = get_dicom_data(self.predict_img)
        t1 = time()
        model = self.build_generator()
        weight_path = './pred_weights'
        for weight_file in os.listdir(weight_path):
            print("start model load weight and predict image......")
            model.load_weights(os.path.join(weight_path, weight_file))
            predict_test = model.predict(low_image, batch_size=1, verbose=1)
            predict_test = (predict_test * 4000 - 1000).astype(np.int16)
            # suffix = weight_file.split('_')[4]
            suffix = weight_file.split('_')[-2]
            check_path(os.path.join(self.output_name, 'predict_'+suffix))

            i = 0
            for dicom_file in os.listdir(self.predict_img):
                ds = pydicom.dcmread(os.path.join(self.predict_img, dicom_file))
                ds.PixelData = predict_test[i].tobytes()
                ds.save_as(os.path.join(self.output_name, 'predict_'+suffix, dicom_file))
                i = i + 1


    def test2dicom(self, low_path, sav_folder='./pre_test'):

        model = self.build_generator()
        # weight_path = './pred_weights'
        # weight_file = 'wgan_filter64_generator_322_0.h5'
        weight_path = './weights_demo_wgan_filter64/126'
        weight_file = 'demo_wgan_filter64_generator_15_0.h5'

        model.load_weights(os.path.join(weight_path, weight_file))

        tmp = low_path.split('\\')
        suffix = weight_file.split('_')[-2]
        sav_folder = sav_folder + '/' + tmp[-1] + '/' + 'predict_' + suffix

        low_image = get_dicom_data(low_path)

        predict_test = model.predict(low_image, batch_size=1, verbose=1)
        predict_test = (predict_test * 4000 - 1000).astype(np.int16)
        # suffix = weight_file.split('_')[4]
        check_path(sav_folder)

        i = 0
        for dicom_file in os.listdir(low_path):
            if os.path.splitext(dicom_file)[-1] == '.dcm':
                ds = pydicom.dcmread(os.path.join(low_path, dicom_file))
                ds.PixelData = predict_test[i].tobytes()
                ds.save_as(os.path.join(sav_folder, dicom_file))
            i = i + 1

        # if os.path.exists(label_path):
        #     label = self.get_dicom_file(label_path)
        #     src_loss, pred_loss = self.loss_calc(np.array(low_list), np.array(pred_list), label)
        #     print('src loss is %.7f, pred_loss is %.7f' % (src_loss, pred_loss))

    def loss_calc(self):
        model = self.build_generator()
        weight_path = './pred_weights'
        weight_file = 'wgan_filter64_generator_322_0.h5'
        model.load_weights(os.path.join(weight_path, weight_file))

        img = np.load('valid_data_crop.npy')
        label = np.load('valid_label_crop.npy')
        predict_test = model.predict(img[..., np.newaxis], batch_size=1, verbose=1)
        loss = np.mean(np.abs(predict_test - label[..., np.newaxis]))
        src_loss = np.mean(np.abs(img-label))
        print(src_loss, loss)

    def pred_test_v3(self, sav_folder='./pred_test/Jupiter/'):
        """
        This function can used to test any folder, that contain .dcm files
        """


        ori_root = self.predict_img
        # path = natsorted(os.listdir(ori_root))
        for root, folder, files in natsorted(os.walk(ori_root)):
            if len(files) > 0:
                if os.path.splitext(files[0])[-1] == '.dcm':
                    test_folder = root
                    self.test2dicom(test_folder, sav_folder=sav_folder)
                    print('Finish one dir!')

if __name__ == "__main__":
    # output_name = "result_out/predict"
    # mytest = TestPredict(predict_img="E:\卷叠伪影\XFFS\Jupiter\\2.16.840.1.114492.84191100109195117.19141104133.55400.51")

    # mytest.predict_img_data()

    # mytest = TestPredict(predict_img='E:\卷叠伪影\XFFS4\CT128_without_BPwgt')
    # mytest.pred_test_v3(sav_folder='./pred_test/XFFS4/')
    # mytest.loss_calc()
    mytest = TestPredict(predict_img='E:\卷叠伪影\\artifacts\Aliasing_artifacts_v1\d8')
    mytest.pred_test_v3(sav_folder='./pred_test/tmp/')
