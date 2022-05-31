# ==============================================     TASK 1 IMPLEMENTATION      ============================================== 

#import libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from Util.util_Integral_Image import *
from Data.Load_Dataset import *
from Util.util_Rectangle_Feature import *
from WeakClf.Weak_Classifier import *

training_data_with_ii = []
weights = np.zeros(len(train_data))
errors_haar_like_feature = []

# Initialization of weights 
for x in range (len(train_data)):
    training_data_with_ii.append((get_integral_image(train_data[x][0]),train_data[x][1]))
    if training_data_with_ii[x][1] == 1:
        weights[x] = 1.0 / (2 * positive_data_length)
    else:
        weights[x] = 1.0 / (2 * negative_data_length)

print("\n Shape of training images       : ",training_data_with_ii[0][0].shape)

#Build Features
def build_features(image_shape,ftype):
    """
            Function :  Use sliding window technique to extract features of all possible shapes of a particular type given a particular shape of the image
            
            Arguments:  image_shape: a tuple of form (height, width)
            
            Returns:   an array of tuples. 
                    ->  First element is an array of the rectangle regions which positively contribute to the feature. 
                    ->  Second element is an array of rectangle regions which negatively contributing to the feature.       

    """
    #print("Extracting Features... Please Wait ")
    height, width = image_shape
    features = []
    for w in range(1, width+1):
        for h in range(1, height+1):
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    #Type-2x rectangle features
                    current = RectangleRegion(i, j, w, h)
                    right = RectangleRegion(i+w, j, w, h)
                    if (ftype == 'type-2-x'):
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [current]))
                    
                    bottom = RectangleRegion(i, j+h, w, h)
                    
                    #Type-2y rectangle features
                    if (ftype=='type-2-y'):
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([current], [bottom]))
                      
                    right_2 = RectangleRegion(i+2*w, j, w, h)
                        
                    #Type-3x rectangle features
                    if (ftype=='type-3-x'):
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, current]))

                    bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        
                    #Type-3y rectangle features
                    if (ftype=='type-3-y'):
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, current]))

                    right_bottom = RectangleRegion(i+w, j+h, w, h)
                        
                    #Type-4 rectangle features
                    if (ftype=='type-4'):
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [current, right_bottom]))

                    j += 1
                i += 1
        # import pdb; pdb.set_trace()
        return np.array(features, dtype=object)

#Apply Feature
def apply_features(features, training_data_with_integral_image):
    """
            Function :  Maps the extracted features on to the training images
            
            Arguments: 1) features: an array of tuples. This is returned by the "build_features ()" function 
                       2) training_data_with_integral_image: An array of tuples. 
                        ->  First element is the numpy array of shape (m, n) representing the integral image. 
                        ->  Second element is its classification (1 or 0)
            
            Returns:   1) X: A numpy array of shape (len(features), len(training_data_with_integral_image)). Each row represents the value of a single feature for each training images
                       2) y: A numpy array of shape len(training_data_with_integral_image). The ith element is the classification of the ith training example

    """
    #print("Applying Features to images...")
    X = np.zeros((len(features), len(training_data_with_integral_image)))
    y = np.array(list(map(lambda data: data[1], training_data_with_integral_image)))
    i = 0
    for positive_regions, negative_regions in features:
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
        X[i] = list(map(lambda data: feature(data[0]), training_data_with_integral_image))
        i += 1
    #print(X)
    return X, y

#Train weak classifier to display errors
def weak_classifier_error(X,y,features,weights):
    """
            Function :  Finds the optimal thresholds for each weak classifier given current weights.
            
            Arguments:  X: A numpy array of shape (len(features), len(training_data_with_integral_image)). Each row represents the value of a single feature for each training image
                        y: A numpy array of shape len(training_data_with_integral_image). The ith element is the classification of the ith training image
                        features: an array of tuples. This is returned by the "build_features ()" function
                        weights: A numpy array of shape len(training_data_with_integral_image). The ith element is the weight assigned to the ith training image
            
            Returns:   A list of weak classifiers

    """  
    error=0
    Error_individual_clf = []
    temp_info = zip(weights,y)
    positives,negatives = 0,0

    for w, label in temp_info:
        if label == 1:
            positives += w
        else:
            negatives += w

    total_features = X.shape[0]
    for index, feature in enumerate(X):
        #if len(Error_individual_clf) % 1000 == 0 and len(Error_individual_clf) != 0:
        #    print("Trained %d classifiers out of %d" % (len(Error_individual_clf), total_features))

        applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            
        pos_seen, neg_seen = 0, 0
        pos_weights, neg_weights = 0, 0
        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for w, f, label in applied_feature:
            error = min(neg_weights + positives - pos_weights, pos_weights + negatives - neg_weights)
            if error < min_error:
                min_error = error
                best_feature = features[index]
                best_threshold = f
                best_polarity = 1 if pos_seen > neg_seen else -1

            if label == 1:
                pos_seen += 1
                pos_weights += w
            else:
                neg_seen += 1
                neg_weights += w
        clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
        Error_individual_clf.append(clf)
    return Error_individual_clf

#Select best features extracted - the one with least errors would be the best feature
def select_best(classifiers, weights, training_data):
    """
            Function :  Selects the best weak classifier out of many weak classifiers for given weights
            
            Arguments: 1) classifiers: An array of weak classifiers
                       2) training_data: An array of tuples. 
                        ->  First element is the numpy array of shape (m, n) representing the integral image. 
                        ->  Second element is its classification (1 or 0)
            
            Returns:   1) A tuple containing the best classifier, the corresponding error, and the array of its accuracy.
    """
    best_clf, best_error, best_accuracy = None, float('inf'), None
    for clf in classifiers:
        error, accuracy = 0, []
        for data, w in zip(training_data, weights):
            correctness = abs(clf.classify(data[0]) - data[1])
            accuracy.append(correctness)
            error += w * correctness
        error = error / len(training_data)
        if error < best_error:
            best_clf, best_error, best_accuracy = clf, error, accuracy
    return best_clf, best_error, best_accuracy

feature_types = ['type-2-x','type-2-y','type-3-x','type-3-y','type-4']

# Compute errors corresponding to each of the HAAR like features 
for feature_type in feature_types:

    #Build Feature
    features_1 = build_features(training_data_with_ii[0][0].shape, feature_type)
    #Apply Feature
    X, y = apply_features(features_1, training_data_with_ii)

    indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
    X = X[indices]
    features_1 = features_1[indices]
    #print("INdices", indices.shape)
    print("\n For ",feature_type ,"-----> Selected %d potential features" % len(X))

    weak_classifiers_Feature_type = weak_classifier_error(X, y, features_1, weights)
    clf,error,accuracy = select_best(weak_classifiers_Feature_type, weights, training_data_with_ii)
    errors_haar_like_feature.append(error)

    # Take the co-ordinates of the features
    feature_coord, feature_type = \
        haar_like_feature_coord(width=training_data_with_ii[0][0].shape[1], height=training_data_with_ii[0][0].shape[0],
                                feature_type=feature_type)
    #print("Haar Feature Co-ordinates extracted !!")
    # Plot Top 10 features of each type of HAAR feature
    fig, axes = plt.subplots(5, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = train_data[0][0]
        image = draw_haar_like_feature(image, 0, 0,
                                    image.shape[1],
                                    image.shape[0],
                                    [feature_coord[indices[idx]]])
        ax.imshow((image * 255).astype(np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])

    _= fig.suptitle('The most important features are ')
    plt.savefig('{}.png'.format(str(feature_type)))

# Show Results
print("\n DISPLAY ERRORs of WEAK CLASSIFIERS corresponding to each of the 5 HAAR-Like FEATURES ")
print("==========================================================================================")
print("\n","                    Error Type-2-x   =   ", errors_haar_like_feature[0])
print("\n","                    Error Type-2-y   =   ", errors_haar_like_feature[1])
print("\n","                    Error Type-3-x   =   ", errors_haar_like_feature[2])
print("\n","                    Error Type-3-y   =   ", errors_haar_like_feature[3])
print("\n","                    Error Type-4     =   ", errors_haar_like_feature[4],"\n")
