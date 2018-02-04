from __future__ import division
import numpy as np
import os, sys


def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1):
  def normalizeFunc(x):
    r=(x-minVal)*newMaxValue/(maxVal-minVal)+newMinValue
    return r
  return np.frompyfunc(normalizeFunc,1,1)

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
  
for modal in ['img', 'txt']:
    for datasets in ['train', 'test', 'validation']:
        prefix= os.path.join(('joint_'+modal), ('joint_reps/joint_'+modal+'_LAST'), datasets)
        data=np.load(prefix+'/joint_'+modal+'_hidden-00001-of-00001.npy')
        #minVal=np.amin(data)
        #maxVal=np.amax(data)
        #outuFuncXArray=normalize_func(minVal,maxVal,0,1)(data)
		outuFuncXArray=znorm(data)
        dataXArray=outuFuncXArray.astype(float)

        np.save(os.path.join(('ff_'+modal), 'data', datasets, ('joint_'+modal+'_hidden-00001-of-00001.npy')), dataXArray)
