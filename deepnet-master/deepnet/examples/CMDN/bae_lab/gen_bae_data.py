import os, sys
import numpy as np

def main():
  prefix = '.'
  image_path = os.path.join(prefix, 'ff_img/ff_reps/train/hidden3-00001-of-00001.npy')
  text_path = os.path.join(prefix, 'ff_txt/ff_reps/train/hidden3-00001-of-00001.npy')

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

  save_dir = os.path.join('bae_lab/data')
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
    s = os.path.join(prefix, 'ff_img/ff_reps', dataset, 'hidden3-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + image_name)
    os.system('cp ' + s + ' ' + d)
    s = os.path.join(prefix, 'ff_txt/ff_reps', dataset, 'hidden3-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + text_name)
    os.system('cp ' + s + ' ' + d)

  os.system('cp feature/train_lab_data.npy bae_lab/data/')
  os.system('cp feature/test_lab_data.npy bae_lab/data/')
  os.system('cp feature/validation_lab_data.npy bae_lab/data/')

  train_lab=np.load('bae_lab/data/train_lab_data.npy')
  bae_train_lab=np.r_[train_lab,train_lab,train_lab]
  np.save('bae_lab/data/bae_train_lab_data.npy',bae_train_lab)

if __name__ == '__main__':
  main()
