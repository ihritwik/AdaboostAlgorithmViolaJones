"""
    Run this code as -> "python3 Train_ViolaJones.py --number_of_weak_clf 20" 
     -> 20 is the number of weak classifiers given as an input by the user. 
    It must be an integer value. (For example, user may enter 5 or 10 or any integer value as an input.

"""
import numpy as np
import argparse
import pickle
from hshukla_adaboost_viola_jones import *
from FaceDetect import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the file name of the trained model')
    parser.add_argument(
        '--number_of_weak_clf', help='Specify the number of weak classifiers for training Viola Jones Algo', required=True, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print(" \n ==== >>>>> START OF TRAINING <<<<< ==== \n")

    # Parse the arguments recieved from command line
    args = parse_args()

    # Store the number of weak classifers as an input in "weak_classifiers"
    weak_classifiers = args.number_of_weak_clf

    # Start TRAINING the model
    train_viola(weak_classifiers)
    
    print("\n ==== >>>>>  END OF TRAINING  <<<<< ==== \n")
    