import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math
import pandas

from tensorflow.python.ops.nn_impl import weighted_cross_entropy_with_logits

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

NUM_EPOCHS_TF_NET = 10


########################################################################
#                Change Things in This Section
########################################################################
# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"

# Use these to select the activation function for the custom neural network
# NOTE: "sigmoid" was used for all my models that had the best results
ACTIVATION_FUNCTION = "sigmoid"
#ACTIVATION_FUNCTION = "relu"

###################################################
#     Uncomment for 2-Layer Custom Neural Network 
###################################################
NUM_EPOCHS_CUSTOM_NET = 20
LAYERS_FOR_CUSTOM_NN = 2
LR_FOR_CUSTOM_NN = 0.1

###################################################
#     Uncomment for N-Layer Custom Neural Network 
###################################################
#NUM_EPOCHS_CUSTOM_NET = 30
#LAYERS_FOR_CUSTOM_NN = 7
#LR_FOR_CUSTOM_NN = 0.01


########################################################################
#                Code Section
########################################################################

class NeuralNetwork_NLayer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, numOfLayers = 2, activationFunc = "sigmoid"):
        if numOfLayers < 2:
            print("Number of layers must be at least 2")
            return
        self.activationFunc = activationFunc
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.numOfLayers = numOfLayers
        self.weights = np.zeros(self.numOfLayers, dtype=np.ndarray)

        # create weights for input layer
        self.weights[0] = np.random.randn(self.inputSize, self.neuronsPerLayer)
        # create weights for inner layers
        for i in range(1, self.numOfLayers - 1):
            self.weights[i] = np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer)
        # create weights for output layer
        self.weights[self.numOfLayers - 1] = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __activationFunction(self, x):
        if self.activationFunc == "sigmoid":
            return self.__sigmoid(x)
        elif self.activationFunc == "relu":
            return self.__relu(x)

    def __activationPrimeFunction(self, x):
        if self.activationFunc == "sigmoid":
            return self.__sigmoidDerivative(x)
        elif self.activationFunc == "relu":
            return self.__reluDerivative(x)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sigValue = self.__sigmoid(x)
        return sigValue * (1 - sigValue)

    # Activation function.
    def __relu(self, x):
        return np.maximum(0, x)

    # Activation prime function.
    def __reluDerivative(self, x):
        return x > 0

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        for epoch in range(0, epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            # iterate over all the xVals and yVals per epoch
            xBatches = self.__batchGenerator(xVals, mbs)
            yBatches = self.__batchGenerator(yVals, mbs)

            # just keep going until it reaches the end of the batches
            while (True):
                try:
                    xBatch = next(xBatches)
                    yBatch = next(yBatches)

                    layerAdjustments = self.__backward(xBatch, yBatch)
                    for i in range(0, self.numOfLayers):
                        self.weights[i] -= layerAdjustments[i]
                except StopIteration:
                    break

    # Forward pass.
    def __forward(self, input):
        layerOutputs = []
        lastLayerOutput = input
        for i in range(0, self.numOfLayers):
            lastLayerOutput = self.__activationFunction(np.dot(lastLayerOutput, self.weights[i]))
            layerOutputs.append(lastLayerOutput)
        return layerOutputs

    # Backward pass.
    def __backward(self, x, y):
        layerOutputs = self.__forward(x)

        error = np.zeros(self.numOfLayers, dtype=np.ndarray)
        layerDeltas = np.zeros(self.numOfLayers, dtype=np.ndarray)
        layerAdjustments = np.zeros(self.numOfLayers, dtype=np.ndarray)

        error[self.numOfLayers - 1] = layerOutputs[len(layerOutputs) - 1] - y
        for i in range(self.numOfLayers - 1, 0, -1):
            layerDeltas[i] = error[i] * self.__activationPrimeFunction(np.dot(layerOutputs[i - 1], self.weights[i]))
            error[i - 1] = np.dot(layerDeltas[i], np.transpose(self.weights[i]))
            layerAdjustments[i] = np.dot(np.transpose(layerOutputs[i - 1]), layerDeltas[i]) * self.lr
        layerDeltas[0] = error[0] * self.__activationPrimeFunction(np.dot(x, self.weights[0]))
        layerAdjustments[0] = np.dot(np.transpose(x), layerDeltas[0]) * self.lr
        return layerAdjustments

    # Predict.
    def predict(self, xVals):
        layers = self.__forward(xVals)

        outputLayer = layers[len(layers) - 1]

        # make the prediction arrays 1s and 0s
        prediction = np.zeros(outputLayer.shape)
        for i in range(0, len(prediction)):
            idxOfMax = np.argmax(outputLayer[i])
            prediction[i][idxOfMax] = 1

        return prediction


class NeuralNetwork_Keras():
    def __init__(self):
        model = tf.keras.models.Sequential([ 
            tf.keras.layers.Dense(512, activation=tf.nn.sigmoid), 
            tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)
            ])
        lossType = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer='adam', loss=lossType, metrics=['accuracy'])
        self.model = model

    def train(self, xVals, yVals, epochs = 100000):
        #self.model.fit(xVals, yVals, epochs=epochs, batch_size=32, validation_split=0.1)
        self.model.fit(xVals, yVals, epochs=epochs, batch_size=32)

    def predict(self, xVals):
        output = self.model.predict(xVals)

        prediction = np.zeros(output.shape)
        for i in range(0, len(prediction)):
            idxOfMax = np.argmax(output[i])
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
        neuralNet = NeuralNetwork_NLayer(IMAGE_SIZE, NUM_CLASSES, 64, numOfLayers=LAYERS_FOR_CUSTOM_NN, learningRate=LR_FOR_CUSTOM_NN, activationFunc=ACTIVATION_FUNCTION)
        neuralNet.train(xTrain, yTrain, NUM_EPOCHS_CUSTOM_NET)
        return neuralNet
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        kerasNeuralNet = NeuralNetwork_Keras()
        kerasNeuralNet.train(xTrain, yTrain, NUM_EPOCHS_TF_NET)
        return kerasNeuralNet
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
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    if ALGORITHM == "custom_net":
        print(f"  with {LAYERS_FOR_CUSTOM_NN} layers")
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Confusion matrix:")

    confMatrix = np.zeros((yTest.shape[1], yTest.shape[1]), dtype=np.int)
    for i in range(preds.shape[0]):
        confMatrix[np.argmax(preds[i])][np.argmax(yTest[i])] += 1
    rowLabels = ['pred-0', 'pred-1', 'pred-2', 'pred-3', 'pred-4', 'pred-5', 'pred-6', 'pred-7', 'pred-8', 'pred-9']
    colLabels = ['truth-0', 'truth-1', 'truth-2', 'truth-3', 'truth-4', 'truth-5', 'truth-6', 'truth-7', 'truth-8', 'truth-9']
    confMatrixLabeled = pandas.DataFrame(confMatrix, columns=colLabels, index=rowLabels)
    print(confMatrixLabeled)


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()