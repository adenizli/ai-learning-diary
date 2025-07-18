from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

class ModelTraining: 
    uniqueNumber = 42

    # these targets are boolean.
    def trainSGDClassifier(self, trainingSet, targets):
        sgdClf = SGDClassifier(random_state=self.uniqueNumber)
        sgdClf.fit(trainingSet, targets)
        self.model = sgdClf
        self.educatedModelTitle = "SGD Classfier"
        self.trainingSet = trainingSet
        self.targets = targets

    def trainDummyClassifier(self, trainingSet, targets):
        dummyClf = DummyClassifier()
        dummyClf.fit(trainingSet, targets)
        
        self.model = dummyClf
        self.trainingSet = trainingSet
        self.educatedModelTitle = "Dummy Classfier"
        self.targets = targets

    def crossValidationTest(self):
        cvs = cross_val_score(self.model, self.trainingSet, self.targets, cv=3, scoring="accuracy")
        
        # Beautiful cross-validation results display
        print("\n" + "="*60)
        print("🎯 CROSS-VALIDATION ACCURACY RESULTS")
        print("="*60)
        print(self.educatedModelTitle)
        print(f"🔄 Cross-Validation: 3-Fold")
        print(f"📈 Scoring Metric: Accuracy")
        print("-"*60)
        
        for i, score in enumerate(cvs, 1):
            print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
        
        print("-"*60)
        print(f"📈 Mean Accuracy: {cvs.mean():.4f} ({cvs.mean()*100:.2f}%)")
        print(f"📊 Standard Deviation: {cvs.std():.4f} ({cvs.std()*100:.2f}%)")
        print(f"🎯 Min Accuracy: {cvs.min():.4f} ({cvs.min()*100:.2f}%)")
        print(f"🚀 Max Accuracy: {cvs.max():.4f} ({cvs.max()*100:.2f}%)")
        print("="*60)
        
        # Performance evaluation
        mean_acc = cvs.mean()
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

    