import scipy.io as sio
import numpy as np
'''
#after sae


prefix_img='./sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/'
prefix_txt='./sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/'
prefix_lab='./sae_img/data/'

image=np.load(prefix_img+"test/hidden2-00001-of-00001.npy")
text=np.load(prefix_txt+"test/hidden2-00001-of-00001.npy")
lab=np.load(prefix_lab+"test_lab_data.npy")

train_image=np.load(prefix_img+"train/hidden2-00001-of-00001.npy")
train_text=np.load(prefix_txt+"train/hidden2-00001-of-00001.npy")
train_lab=np.load(prefix_lab+"train_lab_data.npy")


val_image=np.load(prefix_img+"validation/hidden2-00001-of-00001.npy")
val_text=np.load(prefix_txt+"validation/hidden2-00001-of-00001.npy")
val_lab=np.load(prefix_lab+"validation_lab_data.npy")
'''

#prefix_img='./joint_img_2/joint_reps/joint_img_LAST/'
#prefix_txt='./joint_txt_2/joint_reps/joint_txt_LAST/'
#prefix_img='./ff_img/ff_reps/'
#prefix_txt='./ff_txt/ff_reps/'
#prefix_img='./sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/'
#prefix_txt='./sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/'
prefix_img='./multimodal_dbn/data/dbn_reps/joint_layer2_generated_text/'
prefix_txt='./multimodal_dbn/data/dbn_reps/text_rbm2_LAST/'
#prefix_img='./bae_once/data/ae_reps/bae_LAST/'
#prefix_txt='./bae_once/data/ae_reps/bae_LAST/'

test_image=np.load(prefix_img+"test/text_hidden2-00001-of-00001.npy")
test_text=np.load(prefix_txt+"test/text_hidden2-00001-of-00001.npy")
#test_image=np.load(prefix_img+"test/image_hidden2-00001-of-00001.npy")
#test_text_lab=np.load(prefix_lab+"test_txt_lab_data.npy")

#train_image=np.load(prefix_img+"train/image_hidden2-00001-of-00001.npy")
#train_text=np.load(prefix_txt+"train/text_hidden2-00001-of-00001.npy")
#train_image=np.load(prefix_img+"train/image_hidden2-00001-of-00001.npy")


#val_image=np.load(prefix_img+"validation/hidden3-00001-of-00001.npy")
#val_text=np.load(prefix_txt+"validation/hidden3-00001-of-00001.npy")
#val_image=np.load(prefix_img+"validation/image_hidden2-00001-of-00001.npy")


#print image.shape
#print text.shape

sio.savemat('mdbn_res/i2t/test_image_data.mat', {'test_image':test_image})
sio.savemat('mdbn_res/i2t/test_text_data.mat', {'test_text':test_text})
#sio.savemat('test_img_lab_data.mat', {'test_img_lab':test_image_lab})
#sio.savemat('test_txt_lab_data.mat', {'test_txt_lab':test_text_lab})

#sio.savemat('dbn_res/train_image_data.mat', {'train_image':train_image})
#sio.savemat('dbn_res/train_text_data.mat', {'train_text':train_text})
#sio.savemat('train_lab_data.mat', {'train_lab':train_lab})

#sio.savemat('ff_res/validation_image_data.mat', {'validation_image':val_image})
#sio.savemat('ff_res/validation_text_data.mat', {'validation_text':val_text})
#sio.savemat('validation_lab_data.mat', {'validation_lab':val_lab})
