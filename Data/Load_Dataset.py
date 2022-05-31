import cv2
import pickle
from hshukla_adaboost_viola_jones import *
from FaceDetect import *
from Util.util_file import *

# =========================================== TRAIN DATA ================================================
positive_train_directory = 'Data/Train/Pos_100_Train/'
negative_train_directory = 'Data/Train/Neg_100_Train/'

#Check if file locations are valid
check_path(positive_train_directory)
check_path(negative_train_directory)

positive_train_images = os.listdir(positive_train_directory)
negative_train_images = os.listdir(negative_train_directory)

#print("Number of Positive (Face) training images      : ", len(positive_train_images))
#print("Number of Negative (Non-Face) training images  : ", len(negative_train_images))

positive_data_length = len(positive_train_images)
negative_data_length = len(negative_train_images)

train_data = []

for each_image in positive_train_images:
    image = cv2.imread(positive_train_directory + each_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20,20))
    train_data.append((image, int(1)))

for each_image in negative_train_images:
    image = cv2.imread(negative_train_directory + each_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20,20))
    train_data.append((image, int(0)))

y = np.array(list(map(lambda value:value[1],train_data)))

#print("\n Train Data Loaded ! \n")

# =========================================== TEST DATA ================================================

positive_test_directory = 'Data/Test/Pos/'
negative_test_directory = 'Data/Test/Neg/'

#Check if file locations are valid
check_path(positive_test_directory)
check_path(negative_test_directory)

positive_test_images = os.listdir(positive_test_directory)
negative_test_images = os.listdir(negative_test_directory)

#print("Number of Positive (Face) test images      : ", len(positive_test_images))
#print("Number of Negative (Non-Face) test images  : ", len(negative_test_images))

positive_data_length_test = len(positive_test_images)
negative_data_length_test = len(negative_test_images)

test_data = []

for each_image in positive_test_images:
    image = cv2.imread(positive_test_directory + each_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20,20))
    test_data.append((image, int(1)))

for each_image in negative_test_images:
    image = cv2.imread(negative_test_directory + each_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20,20))
    test_data.append((image, int(0)))

y_test = np.array(list(map(lambda value:value[1],test_data)))
#print("\n Test Data Loaded ! \n")