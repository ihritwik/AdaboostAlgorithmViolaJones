import numpy as np
import math
import pickle
from PIL import Image
from glob import glob
import os
from sklearn.feature_selection import SelectPercentile, f_classif 
from hshukla_adaboost_viola_jones import *
from Data.Load_Dataset import *
from Util.util_Rectangle_Feature import *
def train_viola(t):
    training = train_data
    clf = ViolaJones(T=t)
    clf.Train(training, 100, 100)
    evaluate(clf, training)
    clf.save(str(t))

def test_viola(filename):    
    test = test_data
    clf = ViolaJones.load(filename)
    evaluate(clf, test)
