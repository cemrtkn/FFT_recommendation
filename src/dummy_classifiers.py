import numpy as np

class Majority_Classifier:
    def __init__(self):
        self.guess = None
        

    def fit(self, train_y):
        unique, frequency = np.unique(train_y, 
                              return_counts = True)
        max_id = frequency.argmax(axis = 0)
        most_freq_label = unique[max_id]
        self.guess = most_freq_label

    def predict(self, x):
        pred_shape = x.shape[0]
        preds = np.full(shape=pred_shape, fill_value=self.guess)
        return preds

class Sampling_Classifier:
    def __init__(self):
        self.distribution = None
        self.labels = None
        

    def fit(self, train_y):
        unique, frequency = np.unique(train_y, 
                              return_counts = True)
        frequency = frequency.astype('float64')
        self.distribution = frequency/frequency.sum()
        self.labels = unique

    def predict(self, x):
        pred_shape = x.shape[0]
        preds = []
        for i in range(pred_shape):
            prediction_one_hot = np.random.multinomial(1, self.distribution)
            prediction_id = prediction_one_hot.argmax(axis = 0)
            preds.append(self.labels[prediction_id])
        

        return np.array(preds)


