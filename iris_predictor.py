"""
Python program that implements knn classification on iris dataset.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import operator
from ipdb import set_trace

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
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    classifier = KNN(X_train, Y_train, X_test, Y_test, k)

    predictions = classifier.predict(X_test.iloc[0])

    idx = 0
    counter = 0

    for x in range(len(X_test)):
        prediction = classifier.predict(X_test.iloc[x])[0]
        actual = Y_test.iloc[x]
        if actual == prediction:
            counter += 1
    accuracy = 100* (counter / float(x))
    print "Model accuracy is: {}" .format(accuracy)
    return accuracy

class KNN(object):
    """
    Implementation of k-nearest neighbors
    algorithm.
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, k):
        self.neighbors = k
        self.x_train = pd.DataFrame(X_train)
        self.x_test = pd.DataFrame(X_test)
        self.y_test = pd.Series(Y_test)
        self.y_train = pd.Series(Y_train)
        self.training_data = pd.concat([self.x_train, self.y_train], axis=1)
        print ("Initialization of KNN done.\n")

    def eucl_dist(self, x1, x2):
        """
        Computes euclidean distance between x1, x2.
        """

        x1 = np.array(x1)
        x2 = np.array(x2)
        square = np.dot(x1 - x2, (x1 - x2).T)
        return np.square(square)

    def minkowski_dist(self, x1, x2)

    def predict(self, test_instance):
        """
        Returns the prediction for a test instance
        """

        # Calculate distances between instance and training data
        distances = {}
        for x in range(len(self.x_train)):
            distance = self.eucl_dist(test_instance, self.x_train.iloc[x])
            distances[x] = distance
        # Sort the distances
        sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
        # Calculate neighbors
        neighbors = []
        for x in range(self.neighbors):
            neighbors.append(sorted_d[x][0])

        class_votes = {}
        for x in range(len(neighbors)):
            response = self.training_data.iloc[neighbors[x]][4]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes, key=operator.itemgetter(1), reverse=True)
        return (sorted_votes[0], neighbors)





if __name__ == "__main__":
    import sys

    k_values = []
    accuracy = []
    for x in range(1, 15):
        k_values.append(x)
        accuracy.append(main(x))


    import matplotlib.pyplot as plt

    plt.plot(k_values, accuracy, "r")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
