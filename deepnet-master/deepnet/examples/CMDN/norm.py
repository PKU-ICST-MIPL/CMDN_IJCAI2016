from __future__ import division
import numpy as np
import os, sys


def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1):
  def normalizeFunc(x):
    r=(x-minVal)*newMaxValue/(maxVal-minVal)+newMinValue
    return r
  return np.frompyfunc(normalizeFunc,1,1)

for modal in ['img', 'txt']:
    for datasets in ['train', 'test', 'validation']:
        prefix= os.path.join(('sae_'+modal), 'data/sae_reps/mnist_autoencoder_relu_2layer_LAST', datasets)
        data=np.load(prefix+'/hidden2-00001-of-00001.npy')
        minVal=np.amin(data)
        maxVal=np.amax(data)

        outuFuncXArray=normalize_func(minVal,maxVal,0,1)(data)
        dataXArray=outuFuncXArray.astype(float)

        np.save(os.path.join(('joint_'+modal), ('data/sae_'+modal), datasets, 'hidden2-00001-of-00001_norm.npy'), dataXArray)
