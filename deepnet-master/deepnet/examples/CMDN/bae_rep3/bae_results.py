import os,sys
import numpy as np
from deepnet.fxeval import *
from deepnet.fx_util import *
from numpy.matlib import repmat

def znorm(inMat):
  col=inMat.shape[0]
  row=inMat.shape[1]
  mean_val=np.mean(inMat, axis=0)
  std_val=np.var(inMat, axis=0)
  mean_val=repmat(mean_val, col, 1)
  std_val=repmat(std_val, col, 1)
  return (inMat-mean_val)/std_val

def train():
  label = np.load('data/test_lab_data.npy')
  dic = {}
  #for nn in [32,64,128,256,512,1024]:
  dic = np.zeros((3,))
  prefix = 'data/ae_reps'
  image = np.load(os.path.join(prefix, 'bae_LAST/test/image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'bae_LAST/test/text_tied_hidden-00001-of-00001.npy'))
  #image = np.load(os.path.join(prefix, 'bae_LAST/test/image_hidden3-00001-of-00001.npy'))
  #text = np.load(os.path.join(prefix, 'bae_LAST/test/text_hidden3-00001-of-00001.npy'))
  image=znorm(image)
  text=znorm(text)
  dic[0] = fx_calc_map_label(image, text, label, k=0, dist_method='COS')
  dic[1] = fx_calc_map_label(text, image, label, k=0, dist_method='COS')
  dic[2] = (dic[0]+dic[1]) / 2
  print dic
  fx_pickle('bae_res.pkl', dic)
'''
def best():
  dic = fx_unpickle('bae_res.pkl')
  best_val = -1
  for nn in [32,64,128,256,512,1024]:
    if best_val < dic[nn][2]:
      best_val = dic[nn][2]
      best_nn = nn
  print best_nn, dic[best_nn]
  return best_nn

def best_dist():
  nn = best()
  prefix = 'rbm_'+str(nn)+'/data/ae_reps'
  image = np.load(os.path.join(prefix, 'bae_LAST/test/image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'bae_LAST/test/text_tied_hidden-00001-of-00001.npy'))
  i2t_dist = -fx_cos_distant(image, text)
  i2t = i2t_dist.argsort()
  t2i_dist = -fx_cos_distant(text, image)
  t2i = t2i_dist.argsort()
  save_name = os.path.join('results', 'bae_LAST.pkl')
  fx_pickle(save_name, {'i2t':i2t, 't2i':t2i})

def best_top():
  nn = best()
  prefix = 'rbm_'+str(nn)+'/data/ae_reps'
  image = np.load(os.path.join(prefix, 'bae_LAST/test/image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'bae_LAST/test/text_tied_hidden-00001-of-00001.npy'))
  i2t = fx_calc_map_nolabel_top(image, text)
  t2i = fx_calc_map_nolabel_top(text, image)
  print i2t, t2i, (i2t+t2i) / 2
'''
if __name__ == '__main__':
  p = sys.argv[1]
  if p == 'train':
    train()
'''
  elif p == 'best':
    #best()
  elif p == 'best_dist':
    #best_dist()
  elif p == 'best_top':
    #best_top()
'''
