# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:29:30 2015

@author: Shelly
"""
#import cv2 ### NOTE: Needs OpenCV 2.4(.10) if CV2-KNN used
import math
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve
from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.cluster import KMeans
try:
   import cPickle as pickle
except:
   import pickle

ABSOLUTE = 0
if ABSOLUTE:
    dataPath = "C:/Users/Shelly/Documents/GitHub/swarm_gesture_control/recognition/data/"
else:
    import os
    dir = os.getcwd()
    dataPath = dir+"/data/"
dataExt = ".pickle"

PLOT = 0
from matplotlib import pyplot as plt
#    from matplotlib import cm
#    from matplotlib.lines import Line2D

CV = 0



# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt





def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





def runSVM( dataSet, dataLabels, label_names, testSet, testLabels, title = "Learning Curves" ):
    dataSet = np.array(dataSet)
    dataLabels = np.array(dataLabels)
    
    print "Fitting classifier to data (with cross-validation)"
    
    if dataSet.ndim==1:
        clf = SVC(C=0.75)
        dataSet = dataSet.reshape(-1, 1)
        testSet = testSet.reshape(-1, 1)
        
    else:
        clf = SVC(C=1.0)
    
#        X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataSet, dataLabels, test_size=0.1, random_state=0)
#        clf.fit(X_train, y_train)
#        print "Accuracy:", clf.score(X_test, y_test)
   
    if PLOT and CV:
        plt.figure()
        plt.subplot(121)
    #    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=[50, 80, 110], cv=10)
    
        # SVC is more expensive so we do a lower number of CV iterations:
    #    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10, test_size=0.2, random_state=0)
        cv = cross_validation.ShuffleSplit(dataSet.shape[0], n_iter=10, test_size=0.2, random_state=0)
    #    estimator = SVC(gamma=0.01)
        plot_learning_curve(clf, title, dataSet, dataLabels, (0.5, 1.01), cv=cv, n_jobs=4)
        
        # Confusion matrix
        # From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
#        # Compute confusion matrix
#        cm = confusion_matrix(y_test, y_pred)
#        np.set_printoptions(precision=2)
#        print('Confusion matrix, without normalization:\n')
#        print(cm)
#        plt.figure()
#        plot_confusion_matrix(cm, label_names)
#        
#        # Normalize the confusion matrix by row (i.e by the number of samples
#        # in each class)
#        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print('Normalized confusion matrix:\n')
#        print(cm_normalized)
#        plt.figure()
#        plot_confusion_matrix(cm_normalized, label_names, title='Normalized confusion matrix')

        plt.show()
    
    
    elif CV:
        # From http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
        scores = cross_validation.cross_val_score(clf, dataSet, dataLabels, cv=10)
        print("XVal Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    #    predicted = cross_validation.cross_val_predict(clf, dataSet, dataLabels, cv=10)
    #    print "Running with predictions"#, predicted
    #    print "New accuracy:", metrics.accuracy_score(dataLabels, predicted)
        
        
        
    ### ----- Test specific points ----- ###
    print "Testing specific points (held out)"
        # Apparently xval doesn't return fitted SVM?
    
    predictions = clf.fit(dataSet, dataLabels).predict(testSet)
#    print "Predictions are:\n", predictions
#    print "Expected:\n", testLabels[:,0]
    
    # Compute confusion matrix - no need to normalize here
    cm = confusion_matrix(testLabels[:,0], predictions)
    np.set_printoptions(precision=2)
    print "Confusion matrix, without normalization:\n", cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print "Normalized CM:\n", cm_normalized
    if PLOT:
#        plt.figure()
        plt.subplot(122)
        plot_confusion_matrix(cm_normalized, label_names)
        plt.show()
    
    print classification_report(testLabels[:,0], predictions, target_names=label_names)
    
#    testLabels = np.append(testLabels, predictions, axis=1)
    return predictions
     



def processFiles(files, vidDict):
        # should these be 1 or (1,1)?
    dataSetL = np.zeros((1,1))  # Dummy entry that needs to be removed later!
    dataSetLR = np.zeros((1,2))
    dataSetR = np.zeros((1,1))
    
    dataLabelsL = np.ones((1,1))  # To differentiate from label for class 0
    dataLabelsLR = np.ones((1,1))
    dataLabelsR = np.ones((1,1))
    
    # Spaced out test points (to simulate robots)
    testSetL = np.zeros((1,1))
    testSetLR = np.zeros((1,2))
    testSetR = np.zeros((1,1))
    
    testLabelsL = np.zeros((1,3))
    testLabelsLR = np.zeros((1,3))
    testLabelsR = np.zeros((1,3))
    
#    dataLenL = []
#    dataLenLR = []
#    dataLenR = []
    
    for classID,fileSet in enumerate(files):
        print "Processing files: ", fileSet  # e.g. B30, B25, etc.
        # Reset these for every separate gesture
        angleSet = []
        angleSetL = np.zeros((1,1))
        angleSetLR = np.zeros((1,2))
        angleSetR = np.zeros((1,1))
        
        
        for filename in fileSet:
            data = []
            dataFile = open(dataPath+filename+dataExt, 'rb')
            while 1:
                try:
                    data.append(pickle.load(dataFile))
                except EOFError:
                    break
            
            dataFile.close()
            
            for i,rects in enumerate(data):
                angles = []
                # Trim too-small rects
                for j,rect in enumerate(rects):
                    x,y,w,h,theta = rect
                    if j==0:
                        largest_area = w*h
                        angles.append(theta)
                    else:
                        area = w*h
                        ratio = (area/largest_area)
                        if ratio>=0.1:
                            angles.append(theta)
    #                    else:
    #                        pass
                
                # Keep only largest two rects
        #        if len(rects)>2: data[i]=data[i,:2]
                angleSet.append(np.array(angles[:2]))
        
    #    data = np.array(data)#.astype(np.float32)
    #    data = data.reshape(( len(data)*len(data[0]),2 ))
#            angleSet = np.array(angleSet)
            
            # Add to cumulative angleSet for each gesture 
            middle = len(angleSet)/2
            sixth = math.ceil( float(len(angleSet))/6 )  # round up so all have 6 points, not 7
            dist = [int(s) for s in filename if s.isdigit()]
            dist = float(''.join(map(str,dist)))/10
            if vidDict[filename]=="R":
                count=5
            else:
                count=0            
            
            for idx,angles in enumerate(angleSet):  # all angles from one gesture e.g. B30
                if len(angles)==2:
                    if idx % sixth == 0:  # Save specific test points
                        testSetLR = np.append(testSetLR, [angles], axis=0)
                        testLabelsLR = np.append(testLabelsLR, np.array([[classID,count,dist]]), axis=0 )
                        if vidDict[filename]=="R":
                            count-=1
                        else:
                            count+=1
                    else:
                        angleSetLR = np.append(angleSetLR, [angles], axis=0)
                    
                    
                elif idx < middle:  # first half
                    if vidDict[filename]=="L":
                        if idx % sixth == 0:  # Save specific test points
                            testSetL = np.append(testSetL, angles)
                            testLabelsL = np.append(testLabelsL, np.array([[classID,count,dist]]), axis=0 )
                            count+=1
                        else:
                            angleSetL = np.append(angleSetL, angles)
                    
                            
                    elif vidDict[filename]=="R":
                        if idx % sixth == 0:  # Save specific test points
                            testSetR = np.append(testSetR, angles)
                            testLabelsR = np.append(testLabelsR, np.array([[classID,count,dist]]), axis=0 )
                            count-=1
                        else:
                            angleSetR = np.append(angleSetR, angles)
                    
                            
                    else:
                        print "File not recognized or incorrectly entered...Data not saved."
                    
                else:  # second half
                    if vidDict[filename]=="L":                    
                        if idx % sixth == 0:  # Save specific test points
                            testSetR = np.append(testSetR, angles)
                            testLabelsR = np.append(testLabelsR, np.array([[classID,count,dist]]), axis=0 )
                            count+=1
                        else:
                            angleSetR = np.append(angleSetR, angles)
                            
                    elif vidDict[filename]=="R":
                        if idx % sixth == 0:  # Save specific test points
                            testSetL = np.append(testSetL, angles)
                            testLabelsL = np.append(testLabelsL, np.array([[classID,count,dist]]), axis=0 )
                            count-=1
                        else:
                            angleSetL = np.append(angleSetL, angles)
                    
                            
                    else:
                        print "File not recognized or incorrectly entered - Data not saved."
                    
                    
#                if idx % third == 0:
#                    # ----- Save test points here!! -----
#                    # Saved separately by distance
#                    if len(angles)==2:
#                        testSetLR = np.append(testSetLR, [angles], axis=0)
#                    elif idx < middle:
#                        testSetL = np.append(testSetLR, [angles], axis=0)
#                    else:
#                        pass
                    
#            print filename, "> generated", count, "test points"
                    
        
        # only happens once per gesture 
        angleSetL = angleSetL[1:]
        angleSetLR = angleSetLR[1:]
        angleSetR = angleSetR[1:]
        
        dataSetL = np.append( dataSetL, angleSetL )
        dataSetLR = np.append( dataSetLR, angleSetLR, axis=0 )
        dataSetR = np.append( dataSetR, angleSetR )
        
        dataLabelsL = np.append( dataLabelsL, np.repeat(classID, len(angleSetL))  )
        dataLabelsLR = np.append( dataLabelsLR, np.repeat(classID, len(angleSetLR))  )
        dataLabelsR = np.append( dataLabelsR, np.repeat(classID, len(angleSetR))  )
        
#        dataLenL.append( len(angleSetL) )
#        dataLenLR.append( len(angleSetLR) )
#        dataLenR.append( len(angleSetR) )
        
    # After all data is collated
    dataSetL = dataSetL[1:]
    dataSetLR = dataSetLR[1:]
    dataSetR = dataSetR[1:]
    
    dataLabelsL = dataLabelsL[1:]
    dataLabelsLR = dataLabelsLR[1:]
    dataLabelsR = dataLabelsR[1:]
    
    testSetL = testSetL[1:]
    testSetLR = testSetLR[1:]
    testSetR = testSetR[1:]
    
    testLabelsL = testLabelsL[1:]
    testLabelsLR = testLabelsLR[1:]
    testLabelsR = testLabelsR[1:]
    
    # CHECK THIS - not using yet
#    test_names = [ ["videoB30m","videoB25m","videoB20m","videoB15m","videoB10m"], 
#                 ["videoD30m","videoD25m","videoD20m","videoD15m","videoD10m"],
#                 ["videoG30m","videoG25m","videoG20m","videoG15m","videoG10m"], 
#                 ["videoM30m","videoM25m","videoM20m","videoM15m","videoM10m"] ]
    
    
    return (dataSetL, dataSetLR, dataSetR, 
    dataLabelsL, dataLabelsLR, dataLabelsR,
    testSetL, testSetLR, testSetR,
    testLabelsL, testLabelsLR, testLabelsR)




def plotCustomCM(testData, title="Test set results"):
    cm = np.zeros((5,6))  # (y,x)
    for testPoint in testData:
        [res,x,y] = testPoint
        yp = [1, 1.5, 2, 2.5, 3]
        fp = range(5)
        y = int( np.interp(y, yp, fp) )
        cm[y,x] += res
    
    print "Results:\n", cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print "As normalized CM:\n", cm_normalized
    
    
    plt.figure()
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(6), np.arange(6), rotation=45)
    plt.yticks(np.arange(5), [1, 1.5, 2, 2.5, 3])
    plt.tight_layout()
    plt.ylabel('Distance from controller')
    plt.xlabel('Position on semicircle')
    
    plt.show()




if __name__ == "__main__":
    plt.close("all")
    
    files = [["videoB30m","videoB25m","videoB20m","videoB15m","videoB10m"], 
             ["videoD30m","videoD25m","videoD20m","videoD15m","videoD10m"],
             ["videoG30m","videoG25m","videoG20m","videoG15m","videoG10m"], 
             ["videoM30m","videoM25m","videoM20m","videoM15m","videoM10m"]]
    
    vidDict = {"videoB30m":"L", "videoB25m":"R", "videoB20m":"R", "videoB15m":"L", "videoB10m":"L",
               "videoD30m":"L", "videoD25m":"R", "videoD20m":"L", "videoD15m":"R", "videoD10m":"L",
               "videoG30m":"R", "videoG25m":"L", "videoG20m":"R", "videoG15m":"L", "videoG10m":"R",
               "videoM30m":"R", "videoM25m":"R", "videoM20m":"L", "videoM15m":"R", "videoM10m":"L",}

    
    (dataSetL, dataSetLR, dataSetR, 
    dataLabelsL, dataLabelsLR, dataLabelsR,
    testSetL, testSetLR, testSetR,
    testLabelsL, testLabelsLR, testLabelsR) = processFiles(files, vidDict)
    
    
    
    # ----- L -----
    print "\nRunning classification for left arm data"
    label_names = ["(B)PQRS","(D)JV","EF(G)","AKL(M)N"]  # No longer iterator
    title = "Learning Curves for left arm data"
    testResultL = runSVM(dataSetL, dataLabelsL, label_names, testSetL, testLabelsL, title)
    
    # ----- LR -----
    print "\nRunning classification for both arm data"
    label_names = ["B","D","G","M"]
    title = "Learning Curves for both arm data"
    testResultLR = runSVM(dataSetLR, dataLabelsLR, label_names, testSetLR, testLabelsLR, title)
    
    # ----- R -----
    print "\nRunning classification for right arm data"
    label_names = ["A(B)CD","ABC(D)","(G)NSV","FJ(M)RY"]
    title = "Learning Curves for right arm data"
    testResultR = runSVM(dataSetR, dataLabelsR, label_names, testSetR, testLabelsR, title)



    testLabels = np.concatenate((testLabelsL, testLabelsLR, testLabelsR))  # , dtype=[('x', int), ('y', float)]
    testResults = np.concatenate((testResultL, testResultLR, testResultR))
#    testLabels.view('i8,i8,i8').sort(order=['f0','f2','f1'], axis=0)
    testData = testLabels.copy()
    testData[:,0] = (testData[:,0]==testResults)  # Which results were correct

    print "Regular structured test points (6 per distance)"
    title = "Test set results"
    plotCustomCM(testData, title)
    
    
#    testIdx = np.random.randint(0,30,5)
    testIdx = np.arange(30)
    np.random.shuffle( testIdx )
    shuffleData = testData[testIdx]
    
    numPoints = 5
    iterations = int(30/numPoints)
    for i in range(iterations):
        pass #do stuff??

# TODO:  sort, split by gesture, test randomized swarms
