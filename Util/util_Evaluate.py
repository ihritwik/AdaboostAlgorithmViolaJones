import time
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

def evaluate(clf, dataset):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0
    
    y_actual = []
    y_predicted = []

    for x, y in dataset:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1
        y_actual.append(y)
        start = time.time()
        
        prediction = clf.classify(x)
        # Append the predicted value of classification in y_hat array
        y_predicted.append(prediction)

        classification_time += time.time() - start
        
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(dataset), correct/len(dataset)))
    print("Average Classification Time: %f" % (classification_time / len(dataset)))
    R_O_C_Curve(y_actual,y_predicted)

def R_O_C_Curve(y,y_hat):

    fpr, tpr, _ = metrics.roc_curve(y, y_hat)
    print("fpr = ",fpr)
    print("tpr = ",tpr)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    #create ROC curve
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()