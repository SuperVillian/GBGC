import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def SVM_classifier_nfold(X, Y, n=10):
    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.seed(42)  
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    print ("Performing 10-fold cross validation...")
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the SVM model
        clf2 = SVC(C=20, gamma=1.5e-04)
        clf2.fit(X[train], np.ravel(Y[train]))
        accuracy=clf2.score(X[test],np.ravel(Y[test]))
        predicted_labels=clf2.predict(X[test])
        test_labels=np.reshape(Y[test],(len(Y[test])))
        error=predicted_labels-test_labels
        # print ("Test Fold "+str(i))
        # print("Accuracy= %.2f%%" % (accuracy*100))
        cvscores.append(accuracy * 100)

    print("Average Accuracy={:.2f}-{:.2f}".format(np.mean(cvscores), 
                                        np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)

def KNN_classifier_nfold(X, Y, n=10, k=1, p=2):
    # n-fold cross validation
    # k-NN classification

    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.seed(40)  
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    # print(idx[:10])
    # print ("Performing {}-fold cross validation...".format(n))
    for i, (train , test) in enumerate(skf.split(X, Y)):
    	 # Fit the KNN model
        neigh = KNeighborsClassifier(n_neighbors=k, p=p)
        neigh.fit(X[train], np.ravel(Y[train]))
        accuracy=neigh.score(X[test],np.ravel(Y[test]))
        predicted_labels=neigh.predict(X[test])
        test_labels=np.reshape(Y[test],(len(Y[test])))
        error=predicted_labels-test_labels
        cvscores.append(accuracy * 100)
    # print(cvscores)
    # print("Average Accuracy={:.2f}\\pm{:.2f}".format(np.mean(cvscores), 
    #                                     np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)

from sklearn.linear_model import LogisticRegression

def Logistic_classifier_nfold(X, Y, n=10):
    # n-fold cross validation
    # k-NN classification
    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.seed(42) 
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    print ("Performing {}-fold cross validation...".format(n))
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the KNN model
        logis = LogisticRegression()
        logis.fit(X[train], np.ravel(Y[train]))
        accuracy=logis.score(X[test], np.ravel(Y[test]))
        predicted_labels=logis.predict(X[test])
        test_labels=np.reshape(Y[test], (len(Y[test])))
        error=predicted_labels-test_labels
        cvscores.append(accuracy * 100)
    # print(cvscores)
    print("Average Accuracy={}+-{}".format(np.mean(cvscores), 
                                        np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)