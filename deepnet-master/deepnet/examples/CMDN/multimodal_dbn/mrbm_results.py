import os,sys
import numpy as np
from deepnet.fxeval import *
from deepnet.fx_util import *

def train():
  label = np.load('data/test_lab_data.npy')
  dic = {}
  #for nn in [32,64,128,256,512,1024]:
  dic = np.zeros((3,))
  prefix = 'data/dbn_reps'
  #image = np.load(os.path.join(prefix, 'joint_rbm_LAST/test/joint_hidden-00001-of-00001.npy'))
  #text = np.load(os.path.join(prefix, 'text_rbm2_LAST/test/text_hidden2-00001-of-00001.npy'))
  #text = np.load(os.path.join(prefix, 'joint_rbm_LAST/test/joint_hidden-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))

  #image = np.load(os.path.join(prefix, 'generated_text/test/text_hidden2-00001-of-00001.npy'))
  #text = np.load(os.path.join(prefix, 'generated_text/test/joint_hidden-00001-of-00001.npy'))

  image = np.load('data/test_img_data.npy')
  text = np.load('data/test_txt_data.npy')

  dic[0] = fx_calc_map_label(image, image, label, k=50, dist_method='COS')
  #text = np.load(os.path.join(prefix, 'joint_rbm_LAST/test/joint_hidden1-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'joint_rbm_LAST/test/joint_hidden2-00001-of-00001.npy'))
  #text = np.load(os.path.join(prefix, 'generated_image/test/image_hidden2-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))

  
  #text = np.load(os.path.join(prefix, 'generated_text/test/joint_hidden-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'generated_image/test/joint_hidden-00001-of-00001.npy'))

  dic[1] = fx_calc_map_label(text, text, label, k=50, dist_method='COS')
  dic[2] = (dic[0]+dic[1]) / 2
  print dic[0]
  print dic[1]
  fx_pickle('mrbm_res.pkl', dic)
    
def best():
  dic = fx_unpickle('mrbm_res.pkl')
  best_val = -1
  #for nn in [32,64,128,256,512,1024]:
  if best_val < dic[2]:
    best_val = dic[2]
  #  best_nn = nn
  print best_nn, dic[best_nn]
  return best_nn
  
def best_dist():
  nn = best()
  prefix = 'rbm_'+str(nn)+'/data/rbm_reps'
  image = np.load(os.path.join(prefix, 'joint_layer2_generated_text/test/text_layer-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_rbm2_LAST/test/text_hidden2-00001-of-00001.npy'))
  i2t_dist = -fx_cos_distant(image, text)
  i2t = i2t_dist.argsort()
  text = np.load(os.path.join(prefix, 'joint_layer2_generated_image/test/image_layer-00001-of-00001.npy'))
  image = np.load(os.path.join(prefix, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))
  t2i_dist = -fx_cos_distant(text, image)
  t2i = t2i_dist.argsort()
  save_name = os.path.join('results', 'mrbm_LAST.pkl')
  fx_pickle(save_name, {'i2t':i2t, 't2i':t2i})
  
def best_top():
  nn = best()
  prefix = 'rbm_'+str(nn)+'/data/rbm_reps'
  image = np.load(os.path.join(prefix, 'joint_layer2_generated_text/test/text_layer-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_rbm2_LAST/test/text_hidden2-00001-of-00001.npy'))
  i2t = fx_calc_map_nolabel_top(image,text)
  text = np.load(os.path.join(prefix, 'joint_layer2_generated_image/test/image_layer-00001-of-00001.npy'))
  image = np.load(os.path.join(prefix, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))
  t2i = fx_calc_map_nolabel_top(text, image)
  print i2t, t2i, (i2t+t2i) / 2

if __name__ == '__main__':
  p = sys.argv[1]
  if p == 'train':
    train()
  elif p == 'best':
    best()
  elif p == 'best_dist':
    best_dist()
  elif p == 'best_top':
    best_top()
