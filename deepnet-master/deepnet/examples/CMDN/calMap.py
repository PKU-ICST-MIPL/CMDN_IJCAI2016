import os,sys
import numpy as np
from deepnet.fxeval import *
from deepnet.fx_util import *
from numpy.matlib import repmat

def znorm(inMat):
  col=inMat.shape[0]
  row=inMat.shape[1]
  mean_val=np.mean(inMat, axis=0)
  std_val=np.std(inMat, axis=0)
  mean_val=repmat(mean_val, col, 1)
  std_val=repmat(std_val, col, 1)
  x = np.argwhere(std_val==0)
  for y in x:
    std_val[y[0],y[1]]=1
  return (inMat-mean_val)/std_val

def train():
  label = np.load('feature/test_lab_data.npy')
  dic = {}
  #for nn in [32,64,128,256,512,1024]:
  dic = np.zeros((3,))
  prefix = 'bae_rep3/data/ae_reps'
  image = np.load(os.path.join(prefix, 'bae_LAST/test/image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'bae_LAST/test/text_tied_hidden-00001-of-00001.npy'))
  image = znorm(image)
  text = znorm(text)
  dic[0] = fx_calc_map_label(image, text, label, k=0, dist_method='COS')
  dic[1] = fx_calc_map_label(text, image, label, k=0, dist_method='COS')
  dic[2] = (dic[0]+dic[1]) / 2
  print dic
  fx_pickle('bae_res.pkl', dic)

if __name__ == '__main__':
  p = sys.argv[1]
  if p == 'train':
    train()
