import scipy.io as sio
import numpy as np

prefix='./'

data=sio.loadmat(prefix+'wiki_4096_3000.mat')

train_img_data=data['I_tr']
test_img_data=data['I_te']

train_txt_data=data['T_tr']
test_txt_data=data['T_te']

train_lab_data=data['trCatAll']
test_lab_data=data['teCatAll']

train_lab_data = train_lab_data.reshape(2173,)
test_lab_data = test_lab_data.reshape(462,)

np.save(prefix+'train_img_data.npy', train_img_data)
np.save(prefix+'test_img_data.npy', test_img_data)

np.save(prefix+'train_txt_data.npy', train_txt_data)
np.save(prefix+'test_txt_data.npy', test_txt_data)

np.save(prefix+'train_lab_data.npy', train_lab_data)
np.save(prefix+'test_lab_data.npy', test_lab_data)
