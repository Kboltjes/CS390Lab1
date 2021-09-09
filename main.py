import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"

NEURONS_PER_LAYER = 512




class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sigValue = self.__sigmoid(x)
        return sigValue * (1 - sigValue)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        for epoch in range(0, epochs):
            print(f"batch {epoch} of {epochs}")
            # iterate over all the xVals and yVals per epoch
            xBatches = self.__batchGenerator(xVals, mbs)
            yBatches = self.__batchGenerator(yVals, mbs)

            # just keep going until it reaches the end of the batches
            try:
                while (True):
                    xBatch = next(xBatches)
                    yBatch = next(yBatches)

                    x = xBatch
                    y = yBatch

                    L1out, L2out = self.__forward(x)

                    L2e = L2out - y
                    L2d = L2e * self.__sigmoidDerivative(np.dot(L1out, self.W2))

                    L1e = np.dot(L2d, np.transpose(self.W2))
                    L1d = L1e * self.__sigmoidDerivative(np.dot(x, self.W1))

                    L1a = np.dot(np.transpose(x), L1d) * self.lr
                    L2a = np.dot(np.transpose(L1out), L2d) * self.lr

                    self.W1 -= L1a
                    self.W2 -= L2a
            except StopIteration:
                pass
    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)

        # make the prediction arrays 1s and 0s
        prediction = np.zeros(layer2.shape)
        for i in range(0, len(prediction)):
            idxOfMax = np.argmax(layer2[i])
            prediction[i][idxOfMax] = 1

        return prediction



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    xTrainP = np.array(xTrain).reshape(xTrain.shape[0], IMAGE_SIZE)
    xTestP = np.array(xTest).reshape(xTest.shape[0], IMAGE_SIZE)

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        #print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        neuralNet = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, NEURONS_PER_LAYER)
        neuralNet.train(xTrain, yTrain, 200)
        return neuralNet
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    print(preds)
    print(data[1])
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()