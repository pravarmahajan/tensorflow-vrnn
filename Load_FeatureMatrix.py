import numpy as np
import random
import math
from scipy import stats
import cPickle
import time

def returnTrainValidationTestData():
    
    statisticalFeatureMatrix = cPickle.load(open('data/trajectoriesStatFeatMatrix_5_5', 'rb'))
    print ('Feature Matrix is Loaded!')

    # Load Trajectory --> Driver mapping
    trajectoryToDriver = {}
    driverToTrajectory = {}
    currentNonTaken = 0
    driverIdtoIdConversion = {}
    with open('data/smallSample_5_5.csv') as f:
      for ln in f:
          ps = ln.split(',')
          if 'ID' in ln:
              continue
          trajs = set()
          d = int(ps[0].replace('D-', ''))
          if d in driverIdtoIdConversion:
            d = driverIdtoIdConversion[d]
          else:
            driverIdtoIdConversion[d] = currentNonTaken
            d = currentNonTaken
            currentNonTaken += 1
          if d in driverToTrajectory:
              trajs = driverToTrajectory[d]
          trajs.add(ps[1])
          driverToTrajectory[d] = trajs 

    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []
    for d in driverToTrajectory:    
      trajs = driverToTrajectory[d]
      for t in trajs:
          r = random.random()
          for m in statisticalFeatureMatrix[t]:              
              arr = np.asarray(m)
              i,j = arr.shape
              if i < 128:
                  continue  
              if r < .7:
                  train_data.append(arr)
                  train_labels.append(d)
              elif r < .8:
                  validation_data.append(arr)
                  validation_labels.append(d)
              else:
                  test_data.append(arr)
                  test_labels.append(d) 

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    validation_data   = np.asarray(validation_data, dtype="float32")
    validation_labels = np.asarray(validation_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    #This is to shuffle the train data, used for having better learning process. 
    train_data, train_labels = shuffle_in_unison(train_data, train_labels) 
    
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

    
if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    train, train_labels, eval, eval_labels, test, test_labels = returnTrainValidationTestData()
    
    #Shape of Train, Eval, and Test: 128x35 matrices. Rows --> Time, Columns --> Feature Vector 
    #You may don't need the train_labels, eval_labels, and test_labels, as you try to implement the Auto-Encoder. 