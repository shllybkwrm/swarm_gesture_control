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
   
    if PLOT:
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
    
    
    else:
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
    print('Confusion matrix, without normalization:\n')
    print(cm)
    if PLOT:
#        plt.figure()
        plt.subplot(122)
        plot_confusion_matrix(cm, label_names)
        plt.show()
    
    print classification_report(testLabels[:,0], predictions, target_names=label_names)
    
#    testLabels = np.append(testLabels, predictions, axis=1)
    return predictions
     



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
                    
                    
                elif idx < middle:
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
                    
#            print filename, "generated", count, "test points"
                    
        
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
    print "Created holdout test sets."
    
    # CHECK THIS - not using yet
#    test_names = [ ["videoB30m","videoB25m","videoB20m","videoB15m","videoB10m"], 
#                 ["videoD30m","videoD25m","videoD20m","videoD15m","videoD10m"],
#                 ["videoG30m","videoG25m","videoG20m","videoG15m","videoG10m"], 
#                 ["videoM30m","videoM25m","videoM20m","videoM15m","videoM10m"] ]
    
    
    # ----- L -----
    print "\nRunning classification for left arm data"
    label_names = ["(B)PQRS","(D)JV","EF(G)","AKL(M)N"]  # No longer iterator
    title = "Learning Curves for left arm data"
#    trainTestSVM1DHoldOut(dataSetL,dataLenL, label_names)
#    trainTestSVM1D(dataSetL, dataLabelsL, dataLenL, label_names)
    testDataL = runSVM(dataSetL, dataLabelsL, label_names, testSetL, testLabelsL, title)
    
    # ----- LR -----
    print "\nRunning classification for both arm data"
    label_names = ["B","D","G","M"]
    title = "Learning Curves for both arm data"
#    trainTestSVM2DHoldOut(dataSetLR,dataLenLR, label_names)
#    trainTestSVM2D(dataSetLR, dataLabelsLR, dataLenLR, label_names)
    testDataLR = runSVM(dataSetLR, dataLabelsLR, label_names, testSetLR, testLabelsLR, title)
    
    # ----- R -----
    print "\nRunning classification for right arm data"
    label_names = ["A(B)CD","ABC(D)","(G)NSV","FJ(M)RY"]
    title = "Learning Curves for right arm data"
#    trainTestSVM1DHoldOut(dataSetR,dataLenR, label_names)
#    trainTestSVM1D(dataSetR, dataLabelsR, dataLenR, label_names)
    testDataR = runSVM(dataSetR, dataLabelsR, label_names, testSetR, testLabelsR, title)


    testLabels = np.concatenate((testLabelsL, testLabelsLR, testLabelsR))  # , dtype=[('x', int), ('y', float)]
    testData = np.concatenate((testDataL, testDataLR, testDataR))
#    testData = np.append(testData.reshape(-1, 1), testLabels, axis=1)
#    dtype = [('pred', int), ('act', int), ('x', int), ('y', float)]
#    testData = np.array(testData, dtype=dtype)  # Not working
#    testData.sort(order=['y'])
    testLabels[:,0]=testLabels[:,0]==testData  # Which results were correct

    cm = np.zeros((5,6))  # (y,x)
    for testPoint in testLabels:
        [res,x,y] = testPoint
        yp = [1, 1.5, 2, 2.5, 3]
        fp = range(5)
        y = int( np.interp(y, yp, fp) )
        cm[y,x] += res
    
    print "Results:\n", cm
    
    
    plt.figure()
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Test set results")
    plt.colorbar()
    plt.xticks(np.arange(6), np.arange(6), rotation=45)
    plt.yticks(np.arange(5), [1, 1.5, 2, 2.5, 3])
    plt.tight_layout()
    plt.ylabel('Distance from controller')
    plt.xlabel('Position on semicircle')
    
    plt.show()




#
#def runSVM(dataSet, trainData, trainLabels, testData, testLabels, label_names):
#    
#    # From http://scikit-learn.org/stable/modules/svm.html
#    clf = svm.SVC()
#    print "Fitting SVM to data"
#    
#    
#    clf.fit(trainData, trainLabels)
#    #print "Testing points from B:\n", clf.predict( testData[testLabels==0] )
#    #print "Testing points from D:\n", clf.predict( testData[testLabels==1] )
#    #print "Testing points from G:\n", clf.predict( testData[testLabels==2] )
#    #print "Testing points from M:\n", clf.predict( testData[testLabels==3] )
#    
#    for idx,_ in enumerate(dataSet):
#        print "Testing points from", label_names.next()
#        expected = np.count_nonzero(testLabels==idx)
#        prediction = clf.predict( testData[testLabels==idx] )
#        result = np.count_nonzero( prediction==idx )
#        print "Expected:", expected, "Result:", result
#        if expected > 0:
#            correct = (float(result)/expected)*100
#            print correct, "% correct"
#            if correct<90.0:
#                print "Mispredictions:", prediction
#        else:
#            print "-->None expected?  Check for errors!!"
#  





#
#def trainTestSVM1DHoldOut(dataSet,dataLen, label = iter(["B","D","G","M"]) ):  # For 1 feature!
#    # Reserve random 10% of data for testing
#    trainLen = []
#    randIdx = []
#    trainData = np.empty((1,1))  # Dummy entry that needs to be removed later
#    testData = np.empty((1,1))
#    trainLabels = [] 
#    testLabels = [] 
#    
#    
#    if PLOT: 
#        fig = plt.figure()
#        ax = fig.add_subplot(1,1,1)
#        colors = iter(cm.rainbow( np.linspace(0, 1, len(dataSet)) ))
#        markers = iter(Line2D.filled_markers)
#        markers.next() # skip plain dot
#    
#    
#    for idx, data in enumerate(dataSet):
#        trainLen.append( np.rint(0.9*len(data)).astype(np.int) )
#        dataIdx = np.arange(len(data))
#        np.random.shuffle(dataIdx)
#        randIdx.append(dataIdx)
#        
#        currentTrainData = data[ randIdx[idx][:trainLen[idx]] ]
#        currentTestData = data[ randIdx[idx][trainLen[idx]:] ]
#        
#    #    trainData.append( np.array(data[ trainIdx[idx][:trainLen[idx]] ], dtype=np.float32) )
#    #    testData.append( np.array(data[ trainIdx[idx][trainLen[idx]:] ], dtype=np.float32) )
#    #    trainData = np.append( trainData, data[ randIdx[idx][:trainLen[idx]] ], axis=0 )
#    #    testData = np.append( testData, data[ randIdx[idx][trainLen[idx]:] ], axis=0 )
#    #    trainData.append( data[ randIdx[idx][:trainLen[idx]] ] )
#    #    testData.append( data[ randIdx[idx][trainLen[idx]:] ] )
#        trainData = np.concatenate(( trainData, currentTrainData ))
#        testData = np.concatenate(( testData, currentTestData ))
#        
#    #    labels.append( idx*np.ones(trainLen[idx], dtype=np.int) )
#    #    labels.append( np.repeat(idx, trainLen[idx]) )
#    #    labels = np.append( labels, [idx]*trainLen[idx], axis=0 )
#    #    labels.append( [idx]*trainLen[idx] )
#        trainLabels = np.concatenate(( trainLabels, np.repeat(idx, trainLen[idx]) ))
#        testLabels = np.concatenate(( testLabels, np.repeat(idx, (dataLen[idx]-trainLen[idx]) ) ))
#        
#        # Plot XY data
#    #    plt.scatter(trainData[idx][:,0], trainData[idx][:,1], color=colors.next() )
#    #    plt.scatter(testData[idx][:,0], testData[idx][:,1], color='black', marker=markers.next() )
#        # ---------- How to plot 2 angles?? ----------
#        if PLOT: 
#            ax.scatter(currentTrainData[:,0], currentTrainData[:,1], label='train '+str(idx), color=colors.next(), alpha=0.75  )
#            ax.scatter(currentTestData[:,0], currentTestData[:,1], label='test '+str(idx), color='black', marker=markers.next(), alpha=0.5 )
#        
#    if PLOT: 
#        ax.legend(scatterpoints=1, loc='lower right', ncol=2, fontsize=10)
#        plt.show()
#    
#    trainData = np.array(trainData[1:])#, dtype=np.float32)
#    testData = np.array(testData[1:])#, dtype=np.float32)
#    trainLabels = np.array(trainLabels)#, dtype=np.int)
#    testLabels = np.array(testLabels)#, dtype=np.int)
#    
#    runSVM(dataSet, trainData, trainLabels, testData, testLabels, label)
#    
#
#
#
#def trainTestSVM2DHoldOut(dataSet,dataLen, label = iter(["B","D","G","M"]) ):  # For 2 features!
#    # Reserve random 10% of data for testing
#    trainLen = []
#    randIdx = []
#    trainData = np.empty((1,2))  # Dummy entry that needs to be removed later
#    testData = np.empty((1,2))
#    trainLabels = [] 
#    testLabels = [] 
#    
#    
#    if PLOT: 
#        fig = plt.figure()
#        ax = fig.add_subplot(1,1,1)
#        colors = iter(cm.rainbow( np.linspace(0, 1, len(dataSet)) ))
#        markers = iter(Line2D.filled_markers)
#        markers.next() # skip plain dot
#    
#    
#    for idx, data in enumerate(dataSet):
#        trainLen.append( np.rint(0.9*len(data)).astype(np.int) )
#        dataIdx = np.arange(len(data))
#        np.random.shuffle(dataIdx)
#        randIdx.append(dataIdx)
#        
#        currentTrainData = data[ randIdx[idx][:trainLen[idx]] ]
#        currentTestData = data[ randIdx[idx][trainLen[idx]:] ]
#        
#    #    trainData.append( np.array(data[ trainIdx[idx][:trainLen[idx]] ], dtype=np.float32) )
#    #    testData.append( np.array(data[ trainIdx[idx][trainLen[idx]:] ], dtype=np.float32) )
#    #    trainData = np.append( trainData, data[ randIdx[idx][:trainLen[idx]] ], axis=0 )
#    #    testData = np.append( testData, data[ randIdx[idx][trainLen[idx]:] ], axis=0 )
#    #    trainData.append( data[ randIdx[idx][:trainLen[idx]] ] )
#    #    testData.append( data[ randIdx[idx][trainLen[idx]:] ] )
#        trainData = np.concatenate(( trainData, currentTrainData ))
#        testData = np.concatenate(( testData, currentTestData ))
#        
#    #    labels.append( idx*np.ones(trainLen[idx], dtype=np.int) )
#    #    labels.append( np.repeat(idx, trainLen[idx]) )
#    #    labels = np.append( labels, [idx]*trainLen[idx], axis=0 )
#    #    labels.append( [idx]*trainLen[idx] )
#        trainLabels = np.concatenate(( trainLabels, np.repeat(idx, trainLen[idx]) ))
#        testLabels = np.concatenate(( testLabels, np.repeat(idx, (dataLen[idx]-trainLen[idx]) ) ))
#        
#        # Plot XY data
#    #    plt.scatter(trainData[idx][:,0], trainData[idx][:,1], color=colors.next() )
#    #    plt.scatter(testData[idx][:,0], testData[idx][:,1], color='black', marker=markers.next() )
#        # ---------- How to plot 2 angles?? ----------
#        if PLOT:
#            ax.scatter(currentTrainData[:,0], currentTrainData[:,1], label='train '+str(idx), color=colors.next(), alpha=0.75  )
#            ax.scatter(currentTestData[:,0], currentTestData[:,1], label='test '+str(idx), color='black', marker=markers.next(), alpha=0.5 )
#        
#    if PLOT: 
#        ax.legend(scatterpoints=1, loc='lower right', ncol=2, fontsize=10)
#        plt.show()
#    
#    trainData = np.array(trainData[1:])#, dtype=np.float32)
#    testData = np.array(testData[1:])#, dtype=np.float32)
#    trainLabels = np.array(trainLabels)#, dtype=np.int)
#    testLabels = np.array(testLabels)#, dtype=np.int)
#    
#    
#    runSVM(dataSet, trainData, trainLabels, testData, testLabels, label)
#    
#  





#for filename in files:
#    print "Processing file for", filename
#    dataFile = open(dataPath+filename+dataExt, 'rb')
#    data = []
#    
#    while 1:
#        try:
#            data.append(pickle.load(dataFile))
#        except EOFError:
#            break
#    
#    dataFile.close()
#    data = np.array(data)#.astype(np.float32)
#    data = data.reshape(( len(data)*len(data[0]),2 ))
#    dataSet.append( data )
#    dataLen.append( len(data) )
#
#print ""
#dataSet = np.array(dataSet)  # May not need
#
#
## Reserve random 10% of data for testing
#trainLen = []
#randIdx = []
#trainData = np.empty((1,2))  # Dummy entry that needs to be removed later
#testData = np.empty((1,2))
#trainLabels = [] 
#testLabels = [] 
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#colors = iter(cm.rainbow( np.linspace(0, 1, len(dataSet)) ))
#markers = iter(Line2D.filled_markers)
#markers.next() # skip plain dot
#
#for idx, data in enumerate(dataSet):
#    trainLen.append( np.rint(0.9*len(data)).astype(np.int) )
#    dataIdx = np.arange(len(data))
#    np.random.shuffle(dataIdx)
#    randIdx.append(dataIdx)
#    
#    currentTrainData = data[ randIdx[idx][:trainLen[idx]] ]
#    currentTestData = data[ randIdx[idx][trainLen[idx]:] ]
#    
##    trainData.append( np.array(data[ trainIdx[idx][:trainLen[idx]] ], dtype=np.float32) )
##    testData.append( np.array(data[ trainIdx[idx][trainLen[idx]:] ], dtype=np.float32) )
##    trainData = np.append( trainData, data[ randIdx[idx][:trainLen[idx]] ], axis=0 )
##    testData = np.append( testData, data[ randIdx[idx][trainLen[idx]:] ], axis=0 )
##    trainData.append( data[ randIdx[idx][:trainLen[idx]] ] )
##    testData.append( data[ randIdx[idx][trainLen[idx]:] ] )
#    trainData = np.concatenate(( trainData, currentTrainData ))
#    testData = np.concatenate(( testData, currentTestData ))
#    
##    labels.append( idx*np.ones(trainLen[idx], dtype=np.int) )
##    labels.append( np.repeat(idx, trainLen[idx]) )
##    labels = np.append( labels, [idx]*trainLen[idx], axis=0 )
##    labels.append( [idx]*trainLen[idx] )
#    trainLabels = np.concatenate(( trainLabels, np.repeat(idx, trainLen[idx]) ))
#    testLabels = np.concatenate(( testLabels, np.repeat(idx, (dataLen[idx]-trainLen[idx]) ) ))
#    
#    # Plot XY data
##    plt.scatter(trainData[idx][:,0], trainData[idx][:,1], color=colors.next() )
##    plt.scatter(testData[idx][:,0], testData[idx][:,1], color='black', marker=markers.next() )
#    ax.scatter(currentTrainData[:,0], currentTrainData[:,1], label='train '+str(idx), color=colors.next(), alpha=0.75  )
#    ax.scatter(currentTestData[:,0], currentTestData[:,1], label='test '+str(idx), color='black', marker=markers.next(), alpha=0.5 )
#    
#ax.legend(scatterpoints=1, loc='lower right', ncol=2, fontsize=10)
#plt.show()
#
#trainData = np.array(trainData[1:])#, dtype=np.float32)
#testData = np.array(testData[1:])#, dtype=np.float32)
#trainLabels = np.array(trainLabels)#, dtype=np.int)
#testLabels = np.array(testLabels)#, dtype=np.int)
#
#
## From http://scikit-learn.org/stable/modules/svm.html
#clf = svm.SVC()
#print "Fitting SVM to data:\n", clf.fit(trainData, trainLabels)
##print "Testing points from B:\n", clf.predict( testData[testLabels==0] )
##print "Testing points from D:\n", clf.predict( testData[testLabels==1] )
##print "Testing points from G:\n", clf.predict( testData[testLabels==2] )
##print "Testing points from M:\n", clf.predict( testData[testLabels==3] )
#label = iter(["B","D","G","M"])
#for idx,_ in enumerate(dataSet):
#    print "Testing points from", label.next()
#    expected = float( np.count_nonzero(testLabels==idx) )
#    result = float( np.count_nonzero( clf.predict( testData[testLabels==idx] )==idx ) )
#    print "Expected:", expected, "Result:", result
#    correct = result/expected
#    print correct, "% correct"



    




#trainData = np.append(trainDataB, trainDataD, axis=0).astype(np.float32)
#labels = np.append(np.zeros(len(trainDataB)), np.ones(len(trainDataD)) ).astype(np.int) # Called 'responses' in KNN lit

## From
## http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html#knn-understanding
#print "\nRunning KNN"
#knn = cv2.KNearest()
#knn.train(trainData,labels)
#
#ret, results, neighbors ,dist = knn.find_nearest(testDataB, 3)
#print "\nTested points from B(0):"
#print "result: expected 0 actual", np.sum(results)
##print "neighbors: ", neighbors
##print "distance: ", dist
#
#ret, results, neighbors ,dist = knn.find_nearest(testDataD, 3)
#print "\nTested points from D(1):"
#print "result: expected", len(testDataD), "actual", np.sum(results)
##print "neighbors: ", neighbors
##print "distance: ", dist



# Use K-Means to find clusters
#data = [dataB,dataD]
#data = np.array(data)
#clt = KMeans(n_clusters = 2)
#clt.fit(data)




#filename = "videoD25m"
#print "Processing file for", filename
#dataFile = open(dataPath+filename+dataExt, 'rb')
#
#data = []
#while 1:
#    try:
#        data.append(pickle.load(dataFile))
#    except EOFError:
#        break
#
#dataFile.close()
#dataD = np.array(data)
#dataD = dataD.reshape((len(dataD)*len(dataD[0]),2)).astype(np.float32)




#testLenB = np.rint(0.9*len(dataB)).astype(np.int)
#testLenD = np.rint(0.9*len(dataD)).astype(np.int)
#testIndB = np.arange(len(dataB))
#testIndD = np.arange(len(dataD))
#np.random.shuffle(testIndB)
#np.random.shuffle(testIndD)
#
#trainDataB = dataB[ testIndB[:testLenB] ]
#trainDataD = dataD[ testIndD[:testLenD] ]
#testDataB = dataB[ testIndB[testLenB:] ]
#testDataD = dataD[ testIndD[testLenD:] ]
#
#plt.scatter(trainDataB[:,0], trainDataB[:,1],80,'r','^')
#plt.scatter(trainDataD[:,0], trainDataD[:,1],80,'b','s')
#plt.scatter(testDataB[:,0], testDataB[:,1],80,'g','^')
#plt.scatter(testDataD[:,0], testDataD[:,1],80,'g','s')