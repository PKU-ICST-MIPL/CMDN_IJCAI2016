import os, sys
import numpy as np

def main():
  prefix = '.'
  image_path = os.path.join(prefix, 'bae_rep2/data/ae_reps/bae_LAST/train/image_tied_hidden-00001-of-00001.npy')
  text_path = os.path.join(prefix, 'bae_rep2/data/ae_reps/bae_LAST/train/text_tied_hidden-00001-of-00001.npy')

  image = np.load(image_path)
  text = np.load(text_path)

  numcases1, dimensions1 = image.shape
  numcases2, dimensions2 = text.shape
  fake_data1 = np.zeros((numcases1 / 3, dimensions1))
  fake_data2 = np.zeros((numcases2 / 3, dimensions2))

  image_part = image[numcases1 / 3 * 2: numcases1, :]
  text_part = text[numcases2 / 3 * 2: numcases2, :]

  train_image = np.r_[image_part,fake_data1,image_part]
  train_text = np.r_[fake_data2,text_part,text_part]
  #train_image = np.r_[image,image,image]
  #train_text = np.r_[text,text,text]
  #train_image = image
  #train_text = text

  all_image = np.r_[image_part,image_part,image_part]
  all_text = np.r_[text_part,text_part,text_part]
  #all_image = image
  #all_text = text

  save_dir = os.path.join('bae_rep3/data')
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
    s = os.path.join(prefix, 'bae_rep2/data/ae_reps/bae_LAST', dataset, 'image_tied_hidden-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + image_name)
    os.system('cp ' + s + ' ' + d)
    s = os.path.join(prefix, 'bae_rep2/data/ae_reps/bae_LAST', dataset, 'text_tied_hidden-00001-of-00001.npy')
    d = os.path.join(save_dir, dataset + text_name)
    os.system('cp ' + s + ' ' + d)

  os.system('cp bae_lab/data/train_lab_data.npy bae_rep3/data/')
  os.system('cp bae_lab/data/test_lab_data.npy bae_rep3/data/')
  os.system('cp bae_lab/data/validation_lab_data.npy bae_rep3/data/')
  os.system('cp bae_lab/data/bae_train_lab_data.npy bae_rep3/data/')


if __name__ == '__main__':
  main()

