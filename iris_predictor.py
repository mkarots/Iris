"""
Python program that implements knn classification on iris dataset.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import operator

def main(k = 5):
    setup = \
    """
    Running Algorithm with k = {}
    """.format(k)
    print(setup)
    k = int(k)
    dataset = pd.read_csv("iris.data.txt", header=None)
    x = dataset.iloc[0:, 0:-1]
    y = dataset.iloc[0:,-1]


    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    print("There are {} instances in the training set and {} instances in the test set".format(len(X_train), len(X_test)))
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    classifier = KNN(X_train_std, Y_train, X_test_std, Y_test, k)
    predictions = classifier.fit()

    idx = 0
    counter = 0

    for item in Y_test.iteritems():
        if item[1] == predictions.loc[idx]:
            counter += 1
    accuracy = float(counter) / len(predictions)
    print(accuracy)

class KNN(object):
    """
    Implementation of k-nearest neighbors
    algorithm.
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, k):
        self.neighbors = k
        self.x_train = pd.DataFrame(X_train)
        self.x_test = pd.DataFrame(X_test)
        self.y_train = Y_train
        self.y_test = Y_test



    def fit(self):
        """
        Fit all the points.
        """
        predictions = np.zeros(len(self.x_test))
        predictions = pd.Series(predictions)
        for idx in self.x_test.index:
            instance = self.x_test.loc[idx]

            eucl_dist = pd.Series(self.eucl_dist(instance))
            preds = pd.concat([eucl_dist, self.y_train.reset_index(drop=True)], ignore_index=True, axis=1)
            sortd = preds.sort_values(by=0)
            first_k = sortd.head(n=self.neighbors)
            predictions.loc[idx] = self.decide(first_k)
        return predictions

    def decide(self, neighbors):
        """
        Majority voting
        """
        class_votes = {}
        for item in neighbors.iloc[:,-1]:
            if item not in class_votes:
                class_votes[item] = 1
            else:
                class_votes[item] += 1
        sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def eucl_dist(self, x1):
        """
        Computes euclidean distance between x1, x2.
        """

        distances = np.sqrt(np.dot(self.x_train, x1))
        return distances



if __name__ == "__main__":
    import sys
    k = sys.argv[1]
    main(k)
