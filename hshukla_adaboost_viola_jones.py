from re import T
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
from Util.util_Rectangle_Feature import *
from WeakClf.Weak_Classifier import *
from Util.util_Integral_Image import *
from Util.util_Evaluate import *
from hshukla_adaboost_viola_jones import *

class ViolaJones:
    def __init__(self,T=10):
        self.T = T
        self.alphas = []
        self.classifiers = []
  
    # Model Training  
    def Train(self, train_data, positive_data_length, negative_data_length):
        """
            Function : Trains the Viola Jones classifier on a set of images stored as numpy array
            
            Arguments: 1) train_data: An array of tuples.
                            ->  First element is the numpy array of shape (m, n) representing the integral image. 
                            ->  Second element is its classification (1 or 0)
                       2) positive_data_length: the number of positive samples in training dataset. 
                       3) negative_data_length: the number of negative samples in training dataset
        """
        weights = np.zeros(len(train_data))
        training_data_with_ii = []
        # Initialization of weights 
        #print (" Initializing Weights !!! \n")
        
        for x in range (len(train_data)):
            training_data_with_ii.append((get_integral_image(train_data[x][0]),train_data[x][1]))
            if training_data_with_ii[x][1] == 1:
                weights[x] = 1.0 / (2 * positive_data_length)
            else:
                weights[x] = 1.0 / (2 * negative_data_length)
        
        print("\n Shape of training images                       : ",training_data_with_ii[0][0].shape)

        #Build Feature
        features2x = self.build_features(training_data_with_ii[0][0].shape)
        #Apply Feature
        X, y = self.apply_features(features2x, training_data_with_ii)

        indices = SelectPercentile(f_classif, percentile=25).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features2x = features2x[indices]
        #print("INdices", indices.shape)
        print("For ALL TYPES of haar feature-----> Selected %d potential features" % len(X))

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.weak_classifier_error(X, y, features2x, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data_with_ii)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.classifiers.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))
  
    # Build Feature
    def build_features(self, image_shape):
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
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))
                        bottom = RectangleRegion(i, j+h, w, h)
                        
                        #Type-2y rectangle features / Vertically Adjacent
                        if j + 2 * h < height: 
                            features.append(([immediate], [bottom]))
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        
                        #Type-3x rectangle features /  Horizontally Adjacent
                        if i + 3 * w < width:
                            features.append(([right], [right_2, immediate]))
                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        
                        #Type-3y rectangle features / Vertically Adjacent
                        if j + 3 * h < height: 
                            features.append(([bottom], [bottom_2, immediate]))
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        
                        #Type-4 rectangle features
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features,dtype=object)
  
    # Apply Feature
    def apply_features(self, features, training_data_with_integral_image):
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
  
    # Train weak classifier to display errors
    def weak_classifier_error(self, X,y,features,weights):
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
    
    # Selecting best classifiers based on minimum error criteria
    def select_best(self, classifiers, weights, training_data):
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
  
    # Compute classification result   
    def classify(self, image):
        """
            Function: Classifies an image

            Arguments: image: A numpy 2D array of shape (m, n) representing the image
        
            Returns: 1 if classification is positive and 0 if classification is negative
        """
        total = 0
        ii = get_integral_image(image)
        for alpha, clf in zip(self.alphas, self.classifiers):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0
    
    # Save the trained model
    def save(self, filename):
        """
            Function: save the trained model
            Arguments: filename: The name of the file 
                        *** filename should be without file extension
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
  
    # Load the model's pickle file 
    @staticmethod
    def load(filename):
        """
            Function: To load the model for training and testing purposes
            Arguments: filename: The name of the file 
                        *** filename should be without file extension
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

