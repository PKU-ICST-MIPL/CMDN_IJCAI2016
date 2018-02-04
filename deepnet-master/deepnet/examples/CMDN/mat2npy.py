import scipy.io as sio
import numpy as np

prefix='./feature/'
#the mat file is formatted as n*d, n is number of data and d is dimension
#the following files have the same setting
data=sio.loadmat(prefix+'train_image_data.mat')
train_img_data=data['train_image']

data=sio.loadmat(prefix+'test_image_data.mat')
test_img_data=data['test_image']

data=sio.loadmat(prefix+'validation_image_data.mat')
validation_img_data=data['validation_image']

data=sio.loadmat(prefix+'train_text_data.mat')
train_txt_data=data['train_text']

data=sio.loadmat(prefix+'test_text_data.mat')
test_txt_data=data['test_text']

data=sio.loadmat(prefix+'validation_text_data.mat')
validation_txt_data=data['validation_text']

#the label file is formatted as n*1, noted that the label is start from 0
data=sio.loadmat(prefix+'train_lab_data.mat')
train_lab_data=data['train_lab']
data=sio.loadmat(prefix+'test_lab_data.mat')
test_lab_data=data['test_lab']
data=sio.loadmat(prefix+'validation_lab_data.mat')
validation_lab_data=data['validation_lab']

#the following number should be adjust to the number of training, testing and validation data
train_lab_data = train_lab_data.reshape(2173,)
test_lab_data = test_lab_data.reshape(462,)
validation_lab_data=validation_lab_data.reshape(231,)

np.save(prefix+'train_img_data.npy', train_img_data)
np.save(prefix+'test_img_data.npy', test_img_data)
np.save(prefix+'validation_img_data.npy', validation_img_data)

np.save(prefix+'train_txt_data.npy', train_txt_data)
np.save(prefix+'test_txt_data.npy', test_txt_data)
np.save(prefix+'validation_img_data.npy', validation_txt_data)

np.save(prefix+'train_lab_data.npy', train_lab_data)
np.save(prefix+'test_lab_data.npy', test_lab_data)
np.save(prefix+'validation_lab_data.npy', validation_lab_data)
