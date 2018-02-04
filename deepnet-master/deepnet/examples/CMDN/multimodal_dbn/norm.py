from __future__ import division
import numpy as np

#prefix="./multimodal_dbn/data/dbn_reps/image_rbm2_LAST/validation/"
prefix="./data/"

data=np.load(prefix+"test_txt_data.npy")
def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1):
  def normalizeFunc(x):
    r=(x-minVal)*newMaxValue/(maxVal-minVal)+newMinValue
    return r
  return np.frompyfunc(normalizeFunc,1,1)

minVal=np.amin(data)
maxVal=np.amax(data)

outuFuncXArray=normalize_func(minVal,maxVal,0,1)(data)
dataXArray=outuFuncXArray.astype(float)

np.save(prefix+"test_txt_data_norm.npy", dataXArray);
