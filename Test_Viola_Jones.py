"""
    Run this code as -> "python3 Test_Viola_Jones.py --model 20"

"""
import numpy as np
import argparse
from FaceDetect import *
import sklearn
from sklearn.feature_selection import SelectPercentile, f_classif
from Util.util_Rectangle_Feature import *
def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the file name of the trained model')
    parser.add_argument(
        '--model', help='path storing the trained model', required=True, type=str)
    args = parser.parse_args()
    return args
#Args filename
if __name__ == '__main__':
    # Parse the arguments recieved from command line
    args = parse_args()

    # Store the model name as an input in "filename"
    filename = str(args.model)

    # Start testing the model for performance evaluation
    test_viola(filename)
    