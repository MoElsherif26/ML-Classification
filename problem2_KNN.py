import pandas
import numpy
from sklearn.model_selection import train_test_split

diabetesData = pandas.read_csv("diabetes.csv")
print("The diabetes data before nothing \n")
print(diabetesData)

missingValues = diabetesData.isnull().sum()

print("The missing values in the diabetes data per column: \n")
print(missingValues)

if missingValues.any():
    print("the data have missing values \n")
else:
    print("the data have not missing values \n")

dataTypes = diabetesData.dtypes

print("\nColumns types: ")
print(dataTypes)

features = diabetesData.drop("Outcome", axis=1)
target = diabetesData["Outcome"]

print("\nfeatures: ")
print(features)

print("\ntarget: ")
print(target)
print("\n")



def minMaxScale(feature):
    minValue = feature.min()
    maxValue = feature.max()
    scaledFeature = (feature - minValue) / (maxValue - minValue)
    return scaledFeature

def euclideanDistance(x2, x1):
    distance = numpy.sqrt(numpy.sum((x2 - x1) ** 2))
    return distance


featuresScaled = features.apply(minMaxScale)

featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(featuresScaled, target, test_size=0.3)

def knnClassifier(k, featuresTrain, targetTrain, testInstance):
    distances = []
    
    for i, trainInstance in featuresTrain.iterrows():
        distance = euclideanDistance(testInstance, trainInstance)
        distances.append((distance, targetTrain[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    classVotes = {0: 0, 1: 0}
    for distance, i in neighbors:
        weight = 1 / distance
        classVotes[i] += weight
    
    predictedClass = max(classVotes, key=classVotes.get)
    return predictedClass

def evaluateKnn(k, featuresTrain, targetTrain, featuresTest, targetTest):
    correctPredictions = 0
    
    for i, testInstance in featuresTest.iterrows():
        predictedClass = knnClassifier(k, featuresTrain, targetTrain, testInstance)
        if predictedClass == targetTest[i]:
            correctPredictions += 1
    
    totalInstances = len(featuresTest)
    accuracy = correctPredictions / totalInstances * 100
    
    return correctPredictions, totalInstances, accuracy

sumOfAccuracies = 0
for k in [2, 3, 4, 5, 6]:
    correctPredictions, totalInstances, accuracy = evaluateKnn(k, featuresTrain, targetTrain, featuresTest, targetTest)
    sumOfAccuracies += accuracy
    print(f"k value: {k}\nNumber of correctly classified instances: {correctPredictions}\nTotal number of instances: {totalInstances}\nAccuracy: {accuracy}%\n")

print(f"The average accuracy across the all iterations is: {sumOfAccuracies / 5}%")