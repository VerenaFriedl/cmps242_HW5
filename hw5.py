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

import numpy as np

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
    goodHandles = ["HillaryClinton", "realDonaldTrump"]
    for line in fileH.readlines():
        splitline = line.rstrip('\n').split(',')

        # line starts with twitter handle
        if splitline[0] in goodHandles:
            handles.append(splitline[0])
            tweets.append(','.join(splitline[1:]))
        # line is part of the last tweet
        else:
            tweets[len(tweets) - 1] = tweets[len(tweets) - 1] + "\n" + ','.join(splitline)

    # test if data is read in correctly
    # outfileH = open('./train_out.csv','w')
    # outfileH.write(",".join(header) + "\n")
    # for i in range(0, len(handles)):
    #    outfileH.write(handles[i] + "," + tweets[i]+"\n")

    return handles, tweets

handles, tweets = readTweetData("./train.csv")
