from Database import Database
from Visualization import Visualization
from ModelTraining import ModelTraining

# Create database instance
database = Database()

# Create visualization instance
visualization = Visualization()

# Load the MNIST dataset from local files
database.initMNIST()

# Display some basic information about the loaded data
print(f"\n" + "="*50)
print("MNIST Dataset Information:")
print("="*50)
print(f"Training images: {database.trainSet.shape}")
print(f"Training labels: {database.trainLabel.shape}")
print(f"Test images: {database.testSet.shape}")
print(f"Test labels: {database.testLabel.shape}")

# Show the first training image and its label
print(f"\nFirst training image label: {database.trainLabel[0]}")
print("Displaying the first training image...")
print(f"\n" + "="*50)

# Plot the first digit
# visualization.plotDigit(database.trainSet[0])


# Create model training instance
modelTraining = ModelTraining()

# Use Database methods for preprocessing
fiveTrainingTargets = database.get_binary_classification_targets(5, 'train')
trainSetFlattened = database.get_flattened_data('train')

fiveTestTargets = database.get_binary_classification_targets(5, 'test')
testSetFlattened = database.get_flattened_data('test')

modelTraining.initTrainingSet(trainSetFlattened, fiveTrainingTargets)
modelTraining.initTestSet(testSetFlattened, fiveTestTargets)

modelTraining.trainSGDClassifier()
# modelTraining.trainDummyClassifier(trainSetFlattened, fiveTrainingTargets)

##################################
#    simple visual test system   #
##################################
# Test prediction on a single image using Database method
# testImageFlattened = database.get_single_flattened_image('test', 162)
# prediction = modelTraining.model.predict(testImageFlattened)
# print(f"Prediction for test image 160: {prediction[0]}")
# print(f"Actual label for test image 160: {database.testLabel[162]}")

# visualization.plotDigit(database.testSet[162])

modelTraining.initCrossValidationPredictionsForTrainingSet()
modelTraining.initTestPredictions()

#############################################
#   Check accuracy with cross validation    #
#############################################

# Run cross-validation with beautiful output
modelTraining.crossValidationTest()

#############################################
#   Check accuracy with confusion matrix    #
#############################################

# Run test predictions with confusion matrix
modelTraining.confusionMatrixTest()

#############################################
#   Check accuracy with precision & recall   #
#############################################

modelTraining.precisionRecallTest()

#############################################
#          Adjusting the threshold          #
#############################################

modelTraining.plotPrecisionRecallFunctions()
modelTraining.plotPrecisionRecallTradeoff()
# print(modelTraining.getThresholdForPrecision(0.90))

#############################################
#          Plotting the ROC curve           #
#############################################
modelTraining.initROC()
modelTraining.plotROC()

#############################################
#          Plotting the ROC curve           #
#############################################
modelTraining.initROC()
modelTraining.plotROC()




rfcClassifier = ModelTraining()
rfcClassifier.initTrainingSet(trainSetFlattened, fiveTrainingTargets)
rfcClassifier.initTestSet(testSetFlattened, fiveTestTargets)
rfcClassifier.trainRandomForestClassifier()
rfcClassifier.initRandomForestClassifierCrossValidationPredictions()
print(rfcClassifier.crossValidationPredictions)