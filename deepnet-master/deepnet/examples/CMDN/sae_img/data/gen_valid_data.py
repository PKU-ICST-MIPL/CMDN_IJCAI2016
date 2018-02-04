import os
import numpy as np
from deepnet.fx_util import *

data_dir = '../data'

image_data = np.load(os.path.join(data_dir, 'test_img_data.npy'))
text_data = np.load(os.path.join(data_dir, 'test_txt_data.npy'))
label_data = np.load(os.path.join(data_dir, 'test_lab_data.npy'))

numcases = image_data.shape[0]

#indices = np.arange(numcases)

#rng_state = np.random.get_state()
#np.random.set_state(rng_state)
#np.random.shuffle(indices)
#np.save('indices.npy', indices)
indices = np.load('indices.npy')
print image_data.shape
print indices.shape
image_data = image_data[indices]
text_data = text_data[indices]
label_data = label_data[indices]

valid_num = 231

validation_img_data = image_data[:valid_num]
validation_txt_data = text_data[:valid_num]
validation_lab_data = label_data[:valid_num]

np.save(os.path.join(data_dir, 'validation_img_data.npy'), validation_img_data)
np.save(os.path.join(data_dir, 'validation_txt_data.npy'), validation_txt_data)
np.save(os.path.join(data_dir, 'validation_lab_data.npy'), validation_lab_data)

validation_img_data = image_data[valid_num:]
validation_txt_data = text_data[valid_num:]
validation_lab_data = label_data[valid_num:]

np.save(os.path.join(data_dir, 'test_img_data.npy'), validation_img_data)
np.save(os.path.join(data_dir, 'test_txt_data.npy'), validation_txt_data)
np.save(os.path.join(data_dir, 'test_lab_data.npy'), validation_lab_data)

