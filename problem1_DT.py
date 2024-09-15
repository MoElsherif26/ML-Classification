import numpy
import pandas
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

drugData = pandas.read_csv('drug.csv')

print("The drug data before nothing \n")
print(drugData)

missingValues = drugData.isnull().sum()

print("The missing values in the drug data per column: \n")
print(missingValues)

if missingValues.any():
    print("The data have missing values \n")
else:
    print("The data have not missing values \n")

dataTypes = drugData.dtypes

print("\nColumns types: ")
print(dataTypes)

drugData = drugData.dropna()

print("\nThe drug data after removing missing values: ")
print(drugData)

labelEncoder = LabelEncoder()
categoricalColumns = ['Sex', 'BP', 'Cholesterol', 'Drug']
for col in categoricalColumns:
    drugData[col] = labelEncoder.fit_transform(drugData[col])

print("The preprocessed Data: \n")
print(drugData)

features = drugData.drop('Drug', axis=1)
target = drugData['Drug']

print("\nfeatures: ")
print(features)

print("\ntarget: ")
print(target)
print("\n")

maxAccuracy = -100; bestExperiment = 0; bestTrainSize = 0; bestTestSize = 0

for i in range(5):
    featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target, test_size=0.3, random_state=i)
    decisionTreemodel = DecisionTreeClassifier()
    decisionTreemodel.fit(featuresTrain, targetTrain)

    targetPredicted = decisionTreemodel.predict(featuresTest)

    accuracy = accuracy_score(targetTest, targetPredicted)

    if (accuracy > maxAccuracy):
        maxAccuracy = accuracy
        bestTrainSize = len(featuresTrain)
        bestTestSize = len(featuresTest)
        bestExperiment = i + 1

    print(f"Experiment ({i + 1}):", f"Train Size: ({len(featuresTrain)})", f"Test Size: ({len(featuresTest)})", f"Accuracy: ({accuracy})\n")
    print("\n")

print(f"The best decision Tree model is Experiment: ({bestExperiment}):", f"Train Size: ({bestTrainSize})", f"Test Size: ({bestTestSize})",f"Accuracy: ({maxAccuracy})\n")

trainSizes = range(30, 71, 10)
experimentResults = []

for trainSize in trainSizes:
    treeSizes = []; accuracies = []

    for i in range(5):
        featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target, test_size=(100-trainSize)/100, random_state=i)
        decisionTreemodel = DecisionTreeClassifier()
        decisionTreemodel.fit(featuresTrain, targetTrain)
        targetPredicted = decisionTreemodel.predict(featuresTest)

        treeSize = decisionTreemodel.tree_.node_count
        treeSizes.append(treeSize)
        accuracy = accuracy_score(targetTest, targetPredicted)
        accuracies.append(accuracy)

    meanAccuracy = numpy.mean(accuracies)
    maxAccuracy = numpy.max(accuracies)
    minAccuracy = numpy.min(accuracies)

    meanTreeSize = numpy.mean(treeSizes)
    maxTreeSize = numpy.max(treeSizes)
    minTreeSize = numpy.min(treeSizes)

    experimentResults.append({
        'Training Set Size': trainSize,
        'Mean Accuracy': meanAccuracy,
        'Max Accuracy': maxAccuracy,
        'Min Accuracy': minAccuracy,
        'Mean Tree Size': meanTreeSize,
        'Max Tree Size': maxTreeSize,
        'Min Tree Size': minTreeSize
    })

reportDataFrame = pandas.DataFrame(experimentResults)
print(reportDataFrame)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Mean Accuracy'], label='Mean Accuracy')
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Min Accuracy'], label='Max Accuracy')
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Max Accuracy'], label='Min Accuracy')
plt.title('Accuracy against Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Mean Tree Size'], label='Mean Tree Size')
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Min Tree Size'], label='Max Tree Size')
plt.plot(reportDataFrame['Training Set Size'], reportDataFrame['Max Tree Size'], label='Min Tree Size')
plt.title('Tree Size against Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Number of Nodes')
plt.legend()

plt.tight_layout()
plt.show()