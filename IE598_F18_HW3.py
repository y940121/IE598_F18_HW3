#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 22:50:48 2018

@author: yingzhaocheng
"""
from urllib.request import urlopen
import sys
import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform
from math import sqrt
import matplotlib.pyplot as plot


#read data from uci data repository
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urlopen(target_url)

#arrange data into list for labels and list of lists for attributes
xList = []
labels = []

for line in data:
#split on comma
    row = line.decode('utf-8').strip().split(',')
    xList.append(row)
    
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))
    
nrow = len(xList)
ncol = len(xList[1])
    
type = [0]*3
colCounts = []
    
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                   type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                   type[1] += 1
            else:
                   type[2] += 1
        colCounts.append(type)
        type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
                 "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
        sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                     str(types[1]) + '\t\t' + str(types[2]) + "\n")
        iCol += 1
        
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
            "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4
percentBdry = []

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))  
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")    

#run again with 10 equal intervals  
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#The last column contains categorical variables
col = 60
colData = []
for row in xList:
    colData.append(row[col])

unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)    

#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

type = [0]*3
colCounts = []
#generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()



#read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url,header=None, prefix="V")
#print head and tail of data frame
print(rocksVMines.head())
print(rocksVMines.tail())

#print summary of data frame
summary = rocksVMines.describe()
print(summary)

for i in range(208):
#assign color based on "M" or "R" labels
    if rocksVMines.iat[i,60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
#plot rows of data as if they were series data
dataRow = rocksVMines.iloc[i,0:60]
dataRow.plot(color=pcolor)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()


#calculate correlations between real-valued attributes
dataCol2 = rocksVMines.iloc[0:207,1]
dataCol3 = rocksVMines.iloc[0:207,2]

plot.scatter(dataCol2, dataCol3)

plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()

#change the targets to numeric values
target = []
for i in range(208):
#assign 0 or 1 target value based on "M" or "R" labels
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0)
    else:
        target.append(0.0)
     
#plot 35th attribute
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()  

#To improve the visualization, this version dithers the points a little
# and makes them somewhat transparent
target = []
for i in range(208): 
#assign 0 or 1 target value based on "M" or "R" labels
    # and add some dither
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))
    #plot 35th attribute with semi-opaque points
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()     


#calculate correlations between real-valued attributes
# the attribute should be the col
dataCol2 = rocksVMines.iloc[0:207,1]
dataCol3 = rocksVMines.iloc[0:207,2]
dataCol21 = rocksVMines.iloc[0:207,20]

mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataCol2)
for i in range(numElt):
    mean2 += dataCol2[i]/numElt
    mean3 += dataCol3[i]/numElt
    mean21 += dataCol21[i]/numElt

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataCol2[i] - mean2) * (dataCol2[i] - mean2)/numElt
    var3 += (dataCol3[i] - mean3) * (dataCol3[i] - mean3)/numElt
    var21 += (dataCol21[i] - mean21) * (dataCol21[i] - mean21)/numElt
    
corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataCol2[i] - mean2) * \
              (dataCol3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (dataCol2[i] - mean2) * \
               (dataCol21[i] - mean21) / (sqrt(var2*var21) * numElt)    

sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")



#calculate correlations between real-valued attributes
corMat = DataFrame(rocksVMines.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("My name is Zhaocheng Ying")
print("My NetID is: zying4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
















