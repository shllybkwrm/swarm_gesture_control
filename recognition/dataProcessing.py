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
from sklearn.metrics import confusion_matrix#, classification_report
from scipy import stats, misc
import itertools
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
from matplotlib import cm as cm
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





def runSVM( dataSet, dataLabels, label_names, testSet, testLabels, title = "Learning Curves", mode = 1 ):
    """
    Modes:
        1:  no weighting - 1 vote per robot
        2:  weight by confidence
        3:  2 arms -> double weight
        4:  single arm -> less weight depending on choices
    """
    dataSet = np.array(dataSet)
    dataLabels = np.array(dataLabels)
    
    print "Fitting classifier to data"
    
    if dataSet.ndim==1:  # Single arm data
        clf = SVC(C=0.75)
        dataSet = dataSet.reshape(-1, 1)
        testSet = testSet.reshape(-1, 1)
        if mode==4:
            weight = np.zeros((len(label_names)))
            for idx,name in enumerate(label_names):
                weight[idx] = 1.0/len(name)
        else:
            weight = 1
        
    else:  # 2 arm data
        clf = SVC(C=1.0)
        if mode==3:
            weight = 2
        else:
            weight = 1
    
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
    
    # Apparently xval doesn't return fitted SVM?  Fit again
    predictions = clf.fit(dataSet, dataLabels).predict(testSet)
    conf = clf.decision_function(testSet).max(axis=1).reshape(-1,1)  # higher is better
    if mode==1:
        confidence = np.ones(conf.shape)
    elif mode==2:
        confidence = conf
    elif mode==4 and np.ndim(weight)>0:  # ndim should be 1
        confidence = [c*weight[pred] for (c,pred) in zip(conf, predictions)]
    else: # mode 3 or 4 & single weight (2-arm)
        confidence = conf * weight
#    print "Uncertainty (in dist to separator) is:\n", confidence
#    print "Predictions are:\n", predictions
#    print "Expected:\n", testLabels[:,0]
    
    # Compute confusion matrix - no need to normalize here
    cm = confusion_matrix(testLabels[:,0], predictions)
    print "Confusion matrix (without normalization):\n", cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print "Normalized CM:\n", cm_normalized
    if PLOT:
#        plt.figure()
        plt.subplot(122)
        plot_confusion_matrix(cm_normalized, label_names)
        plt.show()
    
#    print classification_report(testLabels[:,0], predictions, target_names=label_names)
    
#    testLabels = np.append(testLabels, predictions, axis=1)
    return np.concatenate((predictions.reshape(-1,1).astype(int), confidence), axis=1)
     



def processFiles(files, vidDict, num_points, mode="structured"):
    """
    Modes:
        structured:  evenly spaced across all data
        random:  as advertised
        semi-random:  same as random (for now)
        constructed:  same as structured but affects plotting
    """
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
            points = np.arange(len(angleSet))
            if mode=="structured" or mode=="constructed":  # evenly select indices
                interval = math.ceil( float(len(angleSet))/ num_points )  # round up so all have 6 (or num_points) points, not 7
                points = points[points%interval==0]
            elif mode=="random" or mode=="semi-random":  # get random indices
                np.random.shuffle( points )
                points = points[:num_points]
            dist = [int(s) for s in filename if s.isdigit()]
            dist = float(''.join(map(str,dist)))/10
            if vidDict[filename]=="R":
                count=num_points-1
            else:
                count=0            
            
            for idx,angles in enumerate(angleSet):  # all angles from one gesture e.g. B30
                if len(angles)==2:
                    if idx in points:  # Save specific test points
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
                        if idx in points:  # Save specific test points
                            testSetL = np.append(testSetL, angles)
                            testLabelsL = np.append(testLabelsL, np.array([[classID,count,dist]]), axis=0 )
                            count+=1
                        else:
                            angleSetL = np.append(angleSetL, angles)
                    
                            
                    elif vidDict[filename]=="R":
                        if idx in points:  # Save specific test points
                            testSetR = np.append(testSetR, angles)
                            testLabelsR = np.append(testLabelsR, np.array([[classID,count,dist]]), axis=0 )
                            count-=1
                        else:
                            angleSetR = np.append(angleSetR, angles)
                    
                            
                    else:
                        print "File not recognized or incorrectly entered...Data not saved."
                    
                else:  # second half
                    if vidDict[filename]=="L":                    
                        if idx in points:  # Save specific test points
                            testSetR = np.append(testSetR, angles)
                            testLabelsR = np.append(testLabelsR, np.array([[classID,count,dist]]), axis=0 )
                            count+=1
                        else:
                            angleSetR = np.append(angleSetR, angles)
                            
                    elif vidDict[filename]=="R":
                        if idx in points:  # Save specific test points
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
    
    # CHECK THIS - not needed?
#    test_names = [ ["videoB30m","videoB25m","videoB20m","videoB15m","videoB10m"], 
#                 ["videoD30m","videoD25m","videoD20m","videoD15m","videoD10m"],
#                 ["videoG30m","videoG25m","videoG20m","videoG15m","videoG10m"], 
#                 ["videoM30m","videoM25m","videoM20m","videoM15m","videoM10m"] ]
    
    
    return (dataSetL, dataSetLR, dataSetR, 
    dataLabelsL, dataLabelsLR, dataLabelsR,
    testSetL, testSetLR, testSetR,
    testLabelsL, testLabelsLR, testLabelsR)




def processResults(testLabels, testResults, num_gestures, points_per_gesture):
    testIdx = np.arange(points_per_gesture)
    np.random.shuffle( testIdx )
#    shuffleData = testData[testIdx]
    
    splitResults = np.zeros((num_gestures, points_per_gesture, len(testData[0]) ))
#    rows = ["Gesture 0","Gesture 1","Gesture 2","Gesture 3"]
#    columns = ["Desired Mean","Actual Mean","t-value","p-value","p < 0.05?"]
#    cell_text = np.zeros((len(rows), len(columns)))
    
    
    for i in range(num_gestures):
        # Split by gesture
        splitResults[i] = testData[ testData[:,0]==i ]
        
        if PLOT:
#            if testMode=="structured":
            print "Results for gesture", i
            plotResultMatrices(splitResults[i], num_dist, num_points, title="Test set results for gesture "+str(i), mode=3)
#                plotVoteChart(splitResults[i], num_gestures, title="Votes for gesture "+str(i), mode=2)
        
        # Hypothesis test to see if there is a significant difference
        # Returns t, two-tailed p-val
#        (t, p) = stats.ttest_1samp(splitResults[i][:,3], i, equal_var = False)
#        cell_text[i] = [i, np.mean(splitResults[i][:,3]), t, p, (p<0.05)]
#        print "Desired mean:", i, "mean result:", np.mean(splitResults[i][:,3])
#        print "T-test:", t,p
        
        # go through all possible swarms for each gesture
#        for swarm in range(num_points):  
        # swarms of same size already chosen so can evenly split
#            shuffleData = splitResults[i][testIdx]
#            swarms = np.array(np.array_split(shuffleData, num_dist))  # Each split should be num_points long
#            for idx,swarm in enumerate(swarms):
#                if PLOT:
#                    title = "Results for swarm "+str(idx)+" of size "+str(num_points)+" for gesture "+str(i)
#                    print title, "\n", swarm
#                    plotResultMatrices(swarm, num_dist, num_points, title=title, mode=3)  # Combine these into subplots!

#    plt.figure()    
#    plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns)
#    plt.show()

    return splitResults




def plotResultMatrices(testData, num_dist, num_points, title="Test set results", mode=2):
    """
    Modes:
        1:  plot only votes without weighting
        2:  both
        3:  plot only weighted votes
    """
    cm = np.zeros((num_dist, num_points))  # (y,x)
    cm_conf = np.zeros((num_dist, num_points))
    y_range = [1, 1.5, 2, 2.5, 3]
    input_range = range(num_dist)
    for testPoint in testData:
        [act,x,y,res,conf] = testPoint
        y = int( np.interp(y, y_range, input_range) )
        if res==act:  # Check for correct recognition
            cm[y,x] += 1
            cm_conf[y,x] += abs(conf)
        else:  # Mark other robots in the swarm (incorrect rec)
            cm_conf[y,x] += -1*abs(conf)
    
    print "CM:\n", cm, "\n"
    print "Confidence CM:\n", cm_conf, "\n"
#    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    print "As normalized CM:\n", cm_normalized
    
    
    plt.figure()
    
    if mode==2:
        plt.subplot(121)
    if mode==1 or mode==2:
                
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # Find range of values & set colorbar range around it
        plt.clim(0, cm.max())
        
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.arange(num_points), np.arange(num_points))
        plt.yticks(np.arange(num_dist), y_range)
        plt.tight_layout()
        plt.ylabel('Distance from controller')
        plt.xlabel('Position on semicircle')
        
    if mode==2:
        plt.subplot(122)
    if mode==2 or mode==3:
        
        plt.imshow(cm_conf, interpolation='nearest', cmap=plt.cm.RdBu)
        # Find range of values & set colorbar range around it
        maxVal = max(cm_conf.max(), cm_conf.min(), key=abs)
        plt.clim(-1*maxVal,maxVal)
        
        plt.title(title+" by confidence")
        plt.colorbar()
        plt.xticks(np.arange(num_points), np.arange(num_points))
        plt.yticks(np.arange(num_dist), y_range)
        plt.tight_layout()
        plt.ylabel('Distance from controller')
        plt.xlabel('Position on semicircle')
    
    plt.show()




def plotVoteChart(gestureData, num_gestures=4, title="Vote chart", mode=2):
    """
    Modes:
        1:  plot only votes without weighting)
        2:  both (may not need 1 or 2 anymore, instead change weighting mode)
        3:  plot only weighted votes
    """
    votes = np.zeros((num_gestures,1))
    weights = np.zeros((num_gestures,1))
    for idx,testPoint in enumerate(gestureData):
        [act,x,y,res,conf] = testPoint
        votes[res] += 1
        weights[res] += abs(conf)
    
    print "Votes:\n", votes, "\nWeighted votes:\n", weights, "\n"
    
    plt.figure()
    
    if mode==2:
        plt.subplot(121)
    if mode==1 or mode==2:
        
        plt.title(title)
        plt.bar(range(num_gestures), votes)
        plt.xticks(np.arange(num_gestures), np.arange(num_gestures))
    #    plt.yticks(np.arange(num_gestures), np.arange(num_gestures))
        plt.tight_layout()
        plt.ylabel('Votes')
        plt.xlabel('Gestures')
    
    if mode==2:
        plt.subplot(122)
    if mode==2 or mode==3:
        
        plt.title("Weighted "+title)
        plt.bar(range(num_gestures), weights)
        plt.xticks(np.arange(num_gestures), np.arange(num_gestures))
    #    plt.yticks(np.arange(num_gestures), np.arange(num_gestures))
        plt.tight_layout()
        plt.ylabel('Weighted votes')
        plt.xlabel('Gestures')
    
    plt.show()




def plotMultiVoteChart(gestureData, num_gestures=4, title="Vote chart", mode=3, flag="default", swarmSizes=[]):
    """
    Modes:
        1:  plot only votes without weighting
        2:  both
        3:  plot only weighted votes
    Flags:
        default:  as advertised
        constructed:  adjusts for different formatting of results
        average:  adjusts for different formatting of results
        noCalc:  input are already weighted votes (only mode 3 should be used)
    """    
    
    
    # From http://matplotlib.org/examples/api/barchart_demo.html
    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 0.99*height, '%1.2f'%height, ha='center', va='bottom')
            
    
    weightSet = []
        
    for gesture in range(num_gestures):
            
        if flag=="noCalc":
            weights = np.array( gestureData )[:,gesture]
            
#            swarmSizes = [3,5,7]
            num_swarms = len(swarmSizes)
            num_points = len(gestureData)
            title2 = title+str(gesture)
            
        else:
            
            print "\n--- Gesture ", gesture, " ---"
            
#            num_points = 1  # ???
            
            if flag=="constructed":
                swarms = gestureData[gesture]
            elif flag=="average":
                swarms = np.array(list(gestureData[gesture])) 
            else:
                swarms = gestureData
                
            num_swarms = len(swarms)
            votes = np.zeros((num_swarms, num_gestures))
            weights = np.zeros((num_swarms, num_gestures))
            errorBar = np.zeros((num_swarms, num_gestures))
            title2 = title+str(gesture)
                
            # w/ new construction, swarms are already divided by gesture
            for swarmID,swarm in enumerate(swarms):
                if flag=="average":
                    swarmSize = len(swarm)
                elif flag!="constructed":
                    swarm = swarm[gesture]
                # Sort by vote
    #            swarm.view('i8,i8,f8,i8,f8').sort(order=['f3'], axis=0)
                    
                for testPoint in swarm:
                    [act,x,y,res,conf] = testPoint
                    votes[swarmID, res] += 1
                    weights[swarmID, res] += abs(conf)
            
            
                if flag!="average":
                # --- Evaluating confidence in overall results ---
                    copy = weights[swarmID].copy()
                    copy.sort()
                    best_idx = np.where( weights[swarmID]==copy[-1] )[0][0]  # Output as tuple of arrays 
                    secondBest_idx = np.where( weights[swarmID]==copy[-2] )[0][0]
                    best = swarm[ swarm[:,3]==best_idx ][:,4]
                    secondBest = swarm[ swarm[:,3]==secondBest_idx ][:,4]
                    
                    diff = weights[swarmID, best_idx] - weights[swarmID, secondBest_idx]
    #                if ( (diff+weights[swarmID, secondBest_idx]) <= 1.5 ) and ( diff<=weights[swarmID, secondBest_idx] ):
                    if ( diff <= weights[swarmID, secondBest_idx] ):
                        errorBar[swarmID, secondBest_idx] = diff
                
                    if len(swarm)>=10:
                        (t, p) = stats.ttest_ind(best, secondBest, equal_var = False)
                        print "t-test for swarm", swarmID, ":", (t,p)
                        
                    else:
                        percentDiff = 100 * weights[swarmID, best_idx] - weights[swarmID, secondBest_idx] / np.mean( [weights[swarmID, best_idx], weights[swarmID, secondBest_idx]] )
                        print "Percent difference for swarm", swarmID, ":", percentDiff
                        
                else:
                # Collect one vote from entire swarm 
                    copy = weights[swarmID].copy()
                    copy.sort()
                    best_idx = np.where( weights[swarmID]==copy[-1] )[0][0]  # Output as tuple of arrays 
                    new_weights = np.zeros((num_gestures))
                    new_weights[best_idx] = copy[-1]
                    weights[swarmID] = new_weights                
                  
                
            
            # Normalize votes
            for swarmID,swarm_votes in enumerate(votes):
                swarm_weights = weights[swarmID]
                total = sum(swarm_votes)
                total_weight = sum(swarm_weights)
                
                for gestID,vote in enumerate(swarm_votes):
                    weight = swarm_weights[gestID]
                    votes[swarmID,gestID] = float(vote)/total
                    weights[swarmID,gestID] = float(weight)/total_weight
                    
                    if flag!="average":
                        err = errorBar[swarmID,gestID]
                        errorBar[swarmID,gestID] = float(err)/total_weight
                
    #            if flag=="average":
    #            # Collect one vote from entire swarm 
                    
            
            
            if flag=="average":
                votes = np.mean(votes, axis=0)
                weights = np.mean(weights, axis=0)
                num_swarms = 1
                
                
                # --- Evaluating confidence in overall results ---
                swarmID = 0
                errorBar = np.zeros((num_gestures))
                
                copy = weights.copy()
                copy.sort()
                best_idx = np.where( weights==copy[-1] )[0][0]  # Output as tuple of arrays
                secondBest_idx = np.where( weights==copy[-2] )[0][0]
                
                diff = weights[best_idx] - weights[secondBest_idx]
    #                if ( (diff+weights[secondBest_idx]) <= 1.5 ) and ( diff<=weights[secondBest_idx] ):
                if ( diff <= weights[secondBest_idx] ):
                    print "Possibly significant difference between top two votes-"
                    errorBar[secondBest_idx] = diff
                    print "Error bar:", errorBar
        
                percentDiff = 100 * weights[best_idx] - weights[secondBest_idx] / np.mean( [weights[best_idx], weights[secondBest_idx]] )
                print "Percent difference for swarm", swarmID, ":", percentDiff
            
            
            weightSet.append(weights)
        
        
        # --- Plotting starts here ---
        
        plt.figure()
        if flag=="noCalc":
            width = 1.0/((num_points*num_swarms)+1)
        else:
            width = 1.0/(num_swarms+1)  # extra space for between gesture groups
        
        if mode==2:
            plt.subplot(121)
        if mode==1 or mode==2:
            print "Votes:\n", votes, "\n"
            
            plt.title(title2)
#            colors = iter(["r", "b", "g"])
            colors = iter(cm.rainbow(np.linspace(0, 1, num_swarms)))
            for idx in range(num_swarms):
                if flag=="constructed":
                    bars = plt.bar(np.arange(num_gestures)+idx*width, votes[idx,:], width=width, color=next(colors), zorder=3, label="swarm size "+str(len(gestureData[gesture][idx])) )
                elif flag=="average":
                    bars = plt.bar(np.arange(num_gestures)+idx*width, votes, width=width, color=next(colors), zorder=3,  label="all swarms of size "+str(swarmSize) )
                elif flag=="noCalc":
                    print "Not set up, use different testing mode instead!"
                else:
                    bars = plt.bar(np.arange(num_gestures)+idx*width, votes[idx,:], width=width, color=next(colors), zorder=3, label="swarm size "+str(len(gestureData[idx][0])) )
                autolabel(bars)
            
            plt.grid(zorder=0)
            plt.xticks(np.arange(num_gestures+width), np.arange(num_gestures))
            plt.yticks(np.arange(0.0, 1.1, 0.1), np.arange(0.0, 1.1, 0.1))
            plt.ylabel('Normalized votes')
            plt.xlabel('Gestures')
            if flag!="constructed":
                plt.legend()
            plt.tight_layout()
        
        if mode==2:
            plt.subplot(122)
        if mode==2 or mode==3:
            print "Weighted votes:\n", weights, "\n"
            
#            plt.title("Weighted "+title2)
            plt.title(title2)
            
            if flag=="noCalc":
                colors = iter(cm.rainbow(np.linspace(0, 1, num_points*num_swarms)))
            else:
                colors = iter(cm.rainbow(np.linspace(0, 1, num_swarms)))
            
            for idx in range(num_swarms):
                if flag=="constructed":
                    bars = plt.bar(np.arange(num_gestures)+idx*width, weights[idx,:], yerr=errorBar[idx,:], width=width, color=next(colors), zorder=3,  label="swarm size "+str(len(gestureData[gesture][idx])) )
                elif flag=="average":
                    bars = plt.bar(np.arange(num_gestures)+idx*width, weights, yerr=errorBar, width=width, color=next(colors), zorder=3,  label="all swarms of size "+str(swarmSize) )
                elif flag=="noCalc":
                    for idx2 in range(num_points):
#                        bars = plt.bar(np.arange(num_gestures)+idx2*width, weights[idx2], width=width, color=next(colors), zorder=3, label="all swarms of size "+str(swarmSizes[idx]) )
                        bars = plt.bar(np.arange(num_gestures)+idx2*width, weights[idx2], width=width, color=next(colors), zorder=3, label="vote mode "+str(idx2) )
                else:
                    bars = plt.bar(np.arange(num_gestures)+idx*width, weights[idx,:], yerr=errorBar[idx,:], width=width, color=next(colors), zorder=3, label="swarm size "+str(len(gestureData[idx][0])) )
                autolabel(bars)
                
            plt.grid(zorder=0)
            plt.xticks(np.arange(num_gestures)+width, np.arange(num_gestures))
            plt.yticks(np.arange(0.0, 1.1, 0.1), np.arange(0.0, 1.1, 0.1))
            plt.ylabel('Normalized votes by weight')
            plt.xlabel('Gestures')
            if flag!="constructed":
                plt.legend()
            plt.tight_layout()
        
        plt.show()
        
    return weightSet




def constructSwarms(gestureData, num_gestures, mode="notrand", new_size=None):#, title="Vote chart", mode=2):
    """
    Modes:
        rand:  robots selected in random order
        notrand [anything else]:  robots are ordered by confidence
    """
#    num_swarms = len(gestureData)
    returnData = []
        
        
    for swarmID,swarm in enumerate(gestureData):
        correct = []
        incorrect = []
        dataByGesture = []
        
        for gesture in range(num_gestures):
            if mode=="allcomb":
#                new_size = 3
                print "Number of combinations =", misc.comb(len(swarm[gesture]), new_size)
                dataByGesture.append( itertools.combinations(swarm[gesture], new_size) )
                # avoid iterator conversion if possible!
                pass
            
            else:
            
                correct = np.array ( [ testPoint for testPoint in swarm[gesture] if testPoint[0]==testPoint[3] ] )  # for [act,x,y,res,conf] in testPoint
                incorrect = np.array( [ testPoint for testPoint in swarm[gesture] if testPoint[0]!=testPoint[3] ] )
    #            for testPoint in swarm[gesture]:
    #                [act,x,y,res,conf] = testPoint
                
                new_size = len(incorrect)
                if mode=="rand":
                    np.random.shuffle( correct )
                    np.random.shuffle( incorrect )
                else:  # sort by conf (inc)
                    correct.view('i8,i8,f8,i8,f8').sort(order=['f4'], axis=0)
                    incorrect.view('i8,i8,f8,i8,f8').sort(order=['f4'], axis=0)
                newSwarms = []
                # print list(itertools.combinations([1,2,3], 2))
                # print list(itertools.product([0,1], repeat=2))
                # temp=list(itertools.product(swarm[gesture], repeat=2))
                for i in range(1, new_size):
    #                ratio = float(i)*0.1
    #                ratio = i * ( len(incorrect)/10 )
                    newSwarms.append( np.concatenate((incorrect[:i], correct[:(new_size-i)])) )
    #                dataByGesture.append( np.concatenate((incorrect[:i], correct[:(new_size-i)])) )
                dataByGesture.append(newSwarms)
#        singleSwarmData.append(singleGestureData)
        returnData.append(dataByGesture)
#        returnData = dataByGesture
        
        # TODO:  Differentiate between original swarms, Account for randomness
        
    
    return returnData
        
#        for idx,swarmVotes in enumerate(votes):
#            total = sum(swarmVotes)
#            total_weight = sum(weights[idx])
#            for idx2,vote in enumerate(swarmVotes):
#                votes[idx,idx2] = float(vote)/total
#                weights[idx,idx2] = float(weights[idx,idx2])/total_weight
#        
#        print "Votes:\n", votes, "\nWeighted votes:\n", weights, "\n"
#        
#        
#        plt.figure()
#        width = 1.0/(num_swarms+1)
#        
#        if mode==2:
#            plt.subplot(121)
#        if mode==1 or mode==2:
#            
#            plt.title(title2)
##            colors = iter(["r", "b", "g"])
#            colors = iter(cm.rainbow(np.linspace(0, 1, num_swarms)))
#            for idx in range(num_swarms):
#                plt.bar(np.arange(num_gestures)+idx*width, votes[idx,:], width=width, color=next(colors))
#            plt.xticks(np.arange(num_gestures+width), np.arange(num_gestures))
#            plt.tight_layout()
#            plt.ylabel('Votes')
#            plt.xlabel('Gestures')
#        
#        if mode==2:
#            plt.subplot(122)
#        if mode==2 or mode==3:
#            
#            plt.title("Weighted "+title2)
#            
#            colors = iter(cm.rainbow(np.linspace(0, 1, num_swarms)))
#            for idx in range(num_swarms):
#                plt.bar(np.arange(num_gestures)+idx*width, weights[idx,:], width=width, color=next(colors))
#                
#            plt.xticks(np.arange(num_gestures)+width, np.arange(num_gestures))
#            plt.tight_layout()
#            plt.ylabel('Weighted votes')
#            plt.xlabel('Gestures')
#        
#        plt.show()



#def averageVotes(testSwarms):
#    
#    for gestID,gestureData in enumerate(testSwarms):
#        sums = np.zeros((len(testSwarms), len(testSwarms)))
#        lens = np.zeros((len(testSwarms), len(testSwarms)))
#        avgs = np.zeros((len(testSwarms), len(testSwarms)))
#        for swarmID,swarmData in enumerate(gestureData):
#            for swarm in swarmData:
#                swarmSum = sum(swarm[-2,:])
#                swarmAvg
#                for testPoint in swarm:
#                    [act,x,y,res,conf] = testPoint





if __name__ == "__main__":
    plt.close("all")
    np.set_printoptions(precision=2)
    
    files = [["videoB30m","videoB25m","videoB20m","videoB15m","videoB10m"], 
             ["videoD30m","videoD25m","videoD20m","videoD15m","videoD10m"],
             ["videoG30m","videoG25m","videoG20m","videoG15m","videoG10m"], 
             ["videoM30m","videoM25m","videoM20m","videoM15m","videoM10m"]]
    
    vidDict = {"videoB30m":"L", "videoB25m":"R", "videoB20m":"R", "videoB15m":"L", "videoB10m":"L",
               "videoD30m":"L", "videoD25m":"R", "videoD20m":"L", "videoD15m":"R", "videoD10m":"L",
               "videoG30m":"R", "videoG25m":"L", "videoG20m":"R", "videoG15m":"L", "videoG10m":"R",
               "videoM30m":"R", "videoM25m":"R", "videoM20m":"L", "videoM15m":"R", "videoM10m":"L",}
    

#    resultSet = np.zeros((len(points), num_gestures, 1, 5))
    resultSet = []
    num_gestures = len(files)  # 4
    num_dist = len(files[0])  # 5
    points = [6,6,6,6]  # Total size will be num*num_dist (5)
    testSizes = [5]  # per each num_points or just once?
    """
    runSVM voting modes:
        1:  no weighting - 1 vote per robot
        2:  weight by confidence
        3:  2 arms -> double weight
        4:  single arm -> less weight depending on num-choices
    """
    voteModes = [1,2,3,4]  # must be same size as points!
    """
    test modes:
        structured:     evenly spaced 
        random:         as advertised 
        semi-random:    same as rand...?
        constructed:    same as struct but used later 
    """
    testMode = "constructed"
    if testMode=="constructed":
        """
        swarm construction modes:
            rand:  sorted randomly
            allcomb:  all combinations of certain size.  Need to send in new_size to constructSwarms when using this!
            anything else:   sorted by confidence
        """
        constructMode = "allcomb"
    
    
    for pointIdx,num_points in enumerate(points):
        print "\n\n----- Running recognition for", num_points, "points per distance -----"
        
        (dataSetL, dataSetLR, dataSetR, 
        dataLabelsL, dataLabelsLR, dataLabelsR,
        testSetL, testSetLR, testSetR,
        testLabelsL, testLabelsLR, testLabelsR) = processFiles(files, vidDict, num_points, mode=testMode)
        
        
        voteMode = voteModes[pointIdx]
        print "------- Using vote mode", voteMode, "-------"
        
        
        # ----- L -----
        print "\nRunning classification for left arm data"
#        label_names = ["(B)PQRS","(D)JV","EF(G)","AKL(M)N"]  # No longer iterator
        label_names = ["BPQRS","DJV","EFG","AKLMN"]
        title = "Learning Curves for left arm data"
        testResultL = runSVM(dataSetL, dataLabelsL, label_names, testSetL, testLabelsL, title=title, mode=voteMode)
        
        # ----- LR -----
        print "\nRunning classification for both arm data"
        label_names = ["B","D","G","M"]
        title = "Learning Curves for both arm data"
        testResultLR = runSVM(dataSetLR, dataLabelsLR, label_names, testSetLR, testLabelsLR, title=title, mode=voteMode)
        
        # ----- R -----
        print "\nRunning classification for right arm data"
#        label_names = ["A(B)CD","ABC(D)","(G)NSV","FJ(M)RY"]
        label_names = ["ABCD","ABCD","GNSV","FJMRY"]
        title = "Learning Curves for right arm data"
        testResultR = runSVM(dataSetR, dataLabelsR, label_names, testSetR, testLabelsR, title=title, mode=voteMode)
    

        
        testLabels = np.concatenate((testLabelsL, testLabelsLR, testLabelsR))  # , dtype=[('x', int), ('y', float)]
        testResults = np.concatenate((testResultL, testResultLR, testResultR))
                
        # Sort by gesture, position, distance
        testData = np.concatenate((testLabels, testResults),axis=1)  # act, pos, dist, res, conf
        testData.view('i8,i8,f8,i8,f8').sort(order=['f0','f2','f1'], axis=0)
    #    testData = testLabels.copy()
    #    testLabels[:,0] = (testLabels[:,0]==testResults[:,0])  # Which results were correct
        
    
        print  "\n", num_points, "test points per distance"
        if PLOT:
            plotResultMatrices(testData, num_dist, num_points, title="Cumulative test set results", mode=2)
    
            
#        print "Saving / sorting results by gesture"
        # Get random order to create swarms
    #    testIdx = np.random.randint(0,30,5)
        points_per_gesture = num_dist*num_points  # num_dist is fixed, num_points is not
        print "Total possible test points (for each gesture):", points_per_gesture
        
        
        
        resultSet.append( processResults(testLabels, testResults, num_gestures, points_per_gesture) )
        
        
    print "\n\n"
        
#    if PLOT:
    if testMode=="structured":
        plotMultiVoteChart(resultSet, num_gestures, title="Votes for gesture ", mode=3)
    
    
    elif testMode=="constructed":
        
        weightSet = [] #np.zeros((len(testSizes), len(points), num_gestures))
        for idx1,swarmSize in enumerate(testSizes):
            print "- Swarm size", swarmSize, "-"
            testSwarms = constructSwarms(resultSet, num_gestures, mode=constructMode, new_size=swarmSize) 
            for idx2,num_points in enumerate(points):
                print "- num points = 5 *", num_points, "-"
                if constructMode!="allcomb":
    #                pass
                    plotMultiVoteChart(testSwarms[idx2], num_gestures, title="Constructed swarm votes for gesture ", mode=3, flag=testMode)
                else:
                    print "Collecting averaged votes"
    #                averageVotes(testSwarms) 
    #                pass
#                    weightSet[idx1,idx2] = plotMultiVoteChart(testSwarms[idx2], num_gestures, title="All combinations of swarm votes for gesture ", mode=3, flag="average") 
                    weightSet.append( plotMultiVoteChart(testSwarms[idx2], num_gestures, title="All combinations of swarm votes for gesture ", mode=3, flag="average") )

#        print "Weights are", weightSet
        plt.close("all")
        # Need to send in swarmSizes when using flag="noCalc"
        plotMultiVoteChart(weightSet, num_gestures, title="All combinations of swarm votes for gesture ", mode=3, flag="noCalc", swarmSizes=testSizes)

#    if PLOT: 
#        for i in range(num_gestures):
#            plotResultMatrices(testSwarms[i][-1], num_dist, num_points, title="Semi-random swarm results for gesture "+str(i), mode=3)



# TODO: account for randomness of swarms?
# TODO: work around saving all poss combinations (memory error) 


