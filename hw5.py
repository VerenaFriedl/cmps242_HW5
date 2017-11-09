#!/usr/bin/env python3

##########
# File: hw5.py
# Authors: Verena Friedl, Jon Akutagawa, Andrew Bailey
# History: created       Nov 9, 2017
#          last changed  Nov 9, 2017
##########

"""

"""

##########
# Imports
##########

import sys
import numpy as np
import math
import random

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

#import nltk
#from nltk.corpus import stopwords


##########
# Data
##########


def readTweetData(filename):
    """
    Read in tweet collection of "HillaryClinton" and "realDonaldTrump". Other twitter handles would break this function.
    :param filename: File to read tweet data from.
    :return: list with handles and list with tweets; in order of file appearance.
    """
    fileH = open(filename, 'r')
    header = fileH.readline().rstrip().split(',')
    handles = []
    tweets = []
    goodHandles = ["HillaryClinton", "realDonaldTrump","none"]
    for line in fileH.readlines():
        splitline = line.rstrip('\n').split(',')

        # line starts with twitter handle
        if splitline[0] in goodHandles:
            handles.append(splitline[0])
            tweets.append(','.join(splitline[1:]))
        # line is part of the last tweet
        else:
            tweets[len(tweets) - 1] = tweets[len(tweets) - 1] + "\n" + ','.join(splitline)

    # write to file to test if data is read in correctly (should be exactly the same as the input file)
    #outfileH = open('./out.csv','w')
    #outfileH.write(",".join(header) + "\n")
    #for i in range(0, len(handles)):
    #   outfileH.write(handles[i] + "," + tweets[i]+"\n")

    return handles, tweets

handles, tweets = readTweetData("./train.csv")
handles_test, tweets_test = readTweetData("./test.csv")


def encodeLabels(list):
    """
    encode labels in 0 and 1 for comparing "HillaryClinton" = 1 to "realDonaldTrump" = 0
    :param list: input labels
    :return: numpy array with 0 and 1
    """
    labels = list
    labels = [int(x == 'HillaryClinton') for x in labels]
    labels = np.asarray(labels)
    return labels

trainLabels = encodeLabels(handles)


"""
# local path for the downloaded nltk data
nltk.data.path.append("/Users/vfriedl/Google Drive/classes/cmps242/nltk_data")
vectorizer = TfidfVectorizer(input='content',stop_words=stopwords.words('english'), decode_error='ignore', norm='l2')
# merge train and test message lists for encoding
trainAndTest = tweets + tweets_test
X_trainAndTest = vectorizer.fit_transform(trainAndTest)
# split encoded train and test data
X = X_trainAndTest[:len(tweets)]
X_test = X_trainAndTest[len(tweets):]
"""