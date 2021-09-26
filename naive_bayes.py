#-------------------------------------------------------------------------
# AUTHOR: William Armstrong
# FILENAME: naive_bayes.py
# SPECIFICATION: Uses the Naive Bayes strategy to classify test data from training data
# FOR: CS 4210- Assignment #2
# TIME SPENT: ~30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB

#reading the training data
#--> add your Python code here
import csv
db = []
X = []
Y = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         #print(row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =

num_attributes = len(db[0]) - 2
valCounter = [1]*num_attributes
valdict = {}

for row in db:
    xRow = []
    for i, value in enumerate(row):
        if (i == 0):
            continue
        if (i == num_attributes+1):
            break
        if value not in valdict:
            valdict[value] = valCounter[i-1]
            valCounter[i-1] = valCounter[i-1] + 1
        xRow.append(valdict[value])
    X.append(xRow)

#print(X)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

for row in db:
    yval = row[num_attributes+1]
    if (yval == 'Yes'):
        Y.append(1)
    else:
        Y.append(2)
    
#print(Y)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here

testdb = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         testdb.append (row)
#print(testdb)
#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]

for row in testdb:
    predRow = []
    for j, val in enumerate(row):
        if (j == 0):
            continue
        if (j == num_attributes + 1):
            break
        predRow.append(valdict[val])
    predicted = clf.predict_proba( [predRow] )[0]
    playTennis = 'Yes'
    confidence = predicted[0]
    if (predicted[0] < 0.5):
        playTennis = 'No'
        confidence = predicted[1]
    
    if confidence >= 0.75:
        print (row[0].ljust(15) + row[1].ljust(15) + row[2].ljust(15) + row[3].ljust(15) + row[4].ljust(15) + playTennis.ljust(15) + str(confidence).ljust(15))
