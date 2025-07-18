from sklearn.linear_model import SGDClassifier


class ModelTraining: 
    uniqueNumber = 42

    # these targets are boolean.
    def trainSGDClassifier(self, trainingSet, targets):
        sgdClf = SGDClassifier(random_state=self.uniqueNumber)
        sgdClf.fit(trainingSet, targets)
        self.model = sgdClf