import numpy as np

train_image1 = np.load("../sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/train/hidden2-00001-of-00001_norm.npy")
train_image2 = np.load("../multimodal_dbn/data/dbn_reps/image_rbm2_LAST/train/image_hidden2-00001-of-00001_norm.npy")

test_image1 = np.load("../sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/test/hidden2-00001-of-00001_norm.npy")
test_image2 = np.load("../multimodal_dbn/data/dbn_reps/image_rbm2_LAST/test/image_hidden2-00001-of-00001_norm.npy")

validation_image1 = np.load("../sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/validation/hidden2-00001-of-00001_norm.npy")
validation_image2 = np.load("../multimodal_dbn/data/dbn_reps/image_rbm2_LAST/validation/image_hidden2-00001-of-00001_norm.npy")

train_text1 = np.load("../sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/train/hidden2-00001-of-00001_norm.npy")
train_text2 = np.load("../multimodal_dbn/data/dbn_reps/text_rbm2_LAST/train/text_hidden2-00001-of-00001_norm.npy")

test_text1 = np.load("../sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/test/hidden2-00001-of-00001_norm.npy")
test_text2 = np.load("../multimodal_dbn/data/dbn_reps/text_rbm2_LAST/test/text_hidden2-00001-of-00001_norm.npy")

validation_text1 = np.load("../sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/validation/hidden2-00001-of-00001_norm.npy")
validation_text2 = np.load("../multimodal_dbn/data/dbn_reps/text_rbm2_LAST/validation/text_hidden2-00001-of-00001_norm.npy")

train_image = np.column_stack((train_image1, train_image2))
test_image = np.column_stack((test_image1, test_image2))
validation_image = np.column_stack((validation_image1, validation_image2))

train_text = np.column_stack((train_text1, train_text2))
test_text = np.column_stack((test_text1, test_text2))
validation_text = np.column_stack((validation_text1, validation_text2))

np.save("data/train_img_data_2.npy", train_image)
np.save("data/test_img_data_2.npy", test_image)
np.save("data/validation_img_data_2.npy", validation_image)
np.save("data/train_txt_data_2.npy", train_text)
np.save("data/test_txt_data_2.npy", test_text)
np.save("data/validation_txt_data_2.npy", validation_text)
