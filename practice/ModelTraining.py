from sklearn.calibration import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

class ModelTraining: 
    uniqueNumber = 42

    def initTrainingSet(self, trainingSet, targets):
        self.trainingSet = trainingSet
        self.trainingTargets = targets

    def initTestSet(self, testSet, targets):
        self.testSet = testSet
        self.testTargets = targets

    # these targets are boolean.
    def trainSGDClassifier(self):
        sgdClf = SGDClassifier(random_state=self.uniqueNumber)
        sgdClf.fit(self.trainingSet, self.trainingTargets)
        self.model = sgdClf
        self.educatedModelTitle = "SGD Classfier"

    def trainDummyClassifier(self):
        dummyClf = DummyClassifier()
        dummyClf.fit(self.trainingSet, self.trainingTargets)
        self.model = dummyClf
        self.educatedModelTitle = "Dummy Classfier"

    def trainRandomForestClassifier(self):
        rfClf = RandomForestClassifier(random_state=self.uniqueNumber)
        rfClf.fit(self.trainingSet, self.trainingTargets)
        self.model = rfClf
        self.educatedModelTitle = "Random Forest Classfier"

    def initRandomForestClassifierCrossValidationPredictions(self):
        crossValidationPredictions = cross_val_predict(self.model, self.trainingSet, self.trainingTargets, cv=3, method="predict_proba")
        self.crossValidationPredictions = crossValidationPredictions[:, 1]
    
    def initCrossValidationPredictionsForTrainingSet(self):
        self.crossValidationPredictions = cross_val_predict(self.model, self.trainingSet, self.trainingTargets, cv=3)

    def initTestPredictions(self):
        self.testPredictions = self.model.predict(self.testSet)

    def crossValidationTest(self):
        # Beautiful cross-validation results display
        print("\n" + "="*60)
        print("🎯 CROSS-VALIDATION ACCURACY RESULTS")
        print("="*60)
        print(self.educatedModelTitle)
        print(f"🔄 Cross-Validation: 3-Fold")
        print(f"📈 Scoring Metric: Accuracy")
        print("-"*60)
        
        for i, score in enumerate(self.crossValidationPredictions, 1):
            print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
        
        print("-"*60)
        print(f"📈 Mean Accuracy: {self.crossValidationPredictions.mean():.4f} ({self.crossValidationPredictions.mean()*100:.2f}%)")
        print(f"📊 Standard Deviation: {self.crossValidationPredictions.std():.4f} ({self.crossValidationPredictions.std()*100:.2f}%)")
        print(f"🎯 Min Accuracy: {self.crossValidationPredictions.min():.4f} ({self.crossValidationPredictions.min()*100:.2f}%)")
        print(f"🚀 Max Accuracy: {self.crossValidationPredictions.max():.4f} ({self.crossValidationPredictions.max()*100:.2f}%)")
        print("="*60)
        
        # Performance evaluation
        mean_acc = self.crossValidationPredictions.mean()
        if mean_acc >= 0.95:
            print("🏆 EXCELLENT! Model performance is outstanding!")
        elif mean_acc >= 0.90:
            print("✅ VERY GOOD! Model performance is very good!")
        elif mean_acc >= 0.80:
            print("👍 GOOD! Model performance is acceptable!")
        elif mean_acc >= 0.70:
            print("⚠️  FAIR! Model performance could be improved!")
        else:
            print("❌ POOR! Model needs significant improvement!")
        print("="*60)

    def confusionMatrixTest(self):
        cm = confusion_matrix(self.testTargets, self.testPredictions)
        # index 0,0 => non-five images which are matched correctly
        # index 0,1 => non five images which are matched as 5s
        # index 1,0 => five images which are matched 5s
        # index 1,1 => five images which are matched non-five
        print(cm)
    
    def precisionRecallTest(self):
        precisionScore = precision_score(self.trainingTargets, self.crossValidationPredictions)
        print("Precision Score: ", precisionScore)
        recallScore = recall_score(self.trainingTargets, self.crossValidationPredictions)
        print("Recall Score: ", recallScore)
        f1Score = f1_score(self.trainingTargets, self.crossValidationPredictions)
        print("F1 Score: ", f1Score)

    def plotPrecisionRecallFunctions(self):
        precisions, recalls, thresholds = precision_recall_curve(self.trainingTargets, self.crossValidationPredictions)
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.show()

    def plotPrecisionRecallTradeoff(self):
        precisions, recalls, thresholds = precision_recall_curve(self.trainingTargets, self.crossValidationPredictions)
        plt.plot(recalls, precisions, "b-", label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    def getThresholdForPrecision(self, expectedPrecision):
        precisions, recalls, thresholds = precision_recall_curve(self.trainingTargets, self.crossValidationPredictions)
        return thresholds[np.argmax(precisions >= expectedPrecision)]
    
    def getThresholdForRecall(self, expectedRecall):
        precisions, recalls, thresholds = precision_recall_curve(self.trainingTargets, self.crossValidationPredictions)
        return thresholds[np.argmax(recalls >= expectedRecall)]
    
    def initROC(self):
        self.fpr, self.tpr, self.thresholds = roc_curve(self.trainingTargets, self.crossValidationPredictions)

    def plotROC(self):
        plt.plot(self.fpr, self.tpr, "b-", label="ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    def plotROCWithAreaUnderCurve(self):
        plt.plot(self.fpr, self.tpr, "b-", label="ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    def getAreaUnderROC(self):
        return roc_auc_score(self.trainingTargets, self.crossValidationPredictions)