import os, sys
import numpy as np

def main():
  prefix = '../'
  image_path = os.path.join(prefix, 'sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/train/hidden2-00001-of-00001.npy')
  text_path = os.path.join(prefix, 'sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST/train/hidden2-00001-of-00001.npy')
  
  image = np.load(image_path)
  text = np.load(text_path)
  
  numcases1, dimensions1 = image.shape
  numcases2, dimensions2 = text.shape
  fake_data1 = np.zeros((numcases1, dimensions1))
  fake_data2 = np.zeros((numcases2, dimensions2))
  
  train_image = np.r_[image,fake_data1,image]
  train_text = np.r_[fake_data2,text,text]
  
  all_image = np.r_[image,image,image]
  all_text = np.r_[text,text,text]
  
  save_dir = os.path.join('data')
  if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    for dataset in ['train', 'validation', 'test', 'train_all']:
      os.mkdir(os.path.join(save_dir, dataset))
  
  image_name = '_img_data.npy'
  text_name = '_txt_data.npy'
  np.save(os.path.join(save_dir, 'train' + image_name), train_image)
  np.save(os.path.join(save_dir, 'train' + text_name), train_text)
  
  np.save(os.path.join(save_dir, 'train_all' + image_name), all_image)
  np.save(os.path.join(save_dir, 'train_all' + text_name), all_text)
  
  for dataset in ['validation', 'test']:
    s = os.path.join(prefix, 'sae_img/data/sae_reps/mnist_autoencoder_relu_2layer_LAST', dataset, 'hidden2-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + image_name)
    os.system('cp ' + s + ' ' + d)
    s = os.path.join(prefix, 'sae_txt/data/sae_reps/mnist_autoencoder_relu_2layer_LAST', dataset, 'hidden2-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + text_name)
    os.system('cp ' + s + ' ' + d)

if __name__ == '__main__':
  main()
  
