'''
BASIC NEURAL NETWORK//
======================
This program will predict the gender of a person given their weight and height.
===============================================================================
V (2021)
'''

# Import libraries
import numpy as np
import time as t
import random as r

# Declare Variables
network = None
bolTrueFalse = True
userConfirmation = ""
userTrainChoice = 0
userName = ""
userData = None
noOfRandom = 0


# Functions that aid in algorithm but are not needed in class.

# The sigmoid function compresses a tuple down to the range of (0,1).
# Massive negative numbers (i.e beyond int size) become ~0 (V.V with massive positive numbers)
def sigmoid(x):
    # Sigmoid function - f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = None
    # Simple the Derivative of sigmoid - f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def calcLoss(y_true, y_pred):
    # Parameters are numpy lists of exact length.
    # This function is the mean squared error equation. It takes the average of all errors, squared.
    # The difference from the desired output and predicted output are calculated. Lower loss = better predictions.
    return ((y_true - y_pred) ** 2).mean()

def greetUser():
    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print("\n░░░░░░░░░░░░░░░░░░░░░░░░░░░ NEURAL NETWORK ░░░░░░░░░░░░░░░░░░░░░░░░░░░")
    print("\nThe program will predict the a person's gender given some data \(ᵔᵕᵔ)/")
    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    t.sleep(1)
    print("Neural Network starting in...\n3")
    t.sleep(1)
    print("2")
    t.sleep(1)
    print("1")
    t.sleep(1)

class NeuralNetwork:

    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # h1, h2 represent the HIDDEN layers in a neural network, o represents output.
        # The explanation to this function can be found above.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0])
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, allTrues):
        counter = 0
        learn_rate = 0.1
        # times is simply a fancy term for the amount of loops you want through the dataset
        times = 1000

        for time in range(times):
            for x, y_true in zip(data, allTrues):
                # For each neuron, do a feedforward function and apply sigmoid for the predictable outcome.
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                # Make it look pretty with sigmoid :)
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w2 * x[1] + self.b1
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                # ////////////////////////////////////////////////////////////////////////////////////////////
                # Calculate the partial derivatives.
                
                #d_L_d represents the partial D of L / partial D of wN
                d_L_d_ypred = -2 * (y_true - y_pred)

                # o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                # ////////////////////////////////////////////////////////////////////////////////////////////
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # Calculate total losses
                if time % 1 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = calcLoss(allTrues, y_preds)
                    print(f"Learning {time}'s time -- loss: {loss}")
                
                # Stop training when reaching under 99%
                #if loss < 0.0001:
                #    return True
                

# Define dataset
data = np.array([
# [0] = Weight in pounds - 135 (arbitrary number to manage data quicker)
# [1] = Height in inches - 66 (arbitrary again)
  [-2, -1],  # W
  [25, 6],   # F
  [17, 4],   # F
  [-15, -6], # W
  
])

all_y_trues = np.array([
# 1 = Female , 0 = Male
  1, # W
  0, # F
  0, # F
  1, # W
])

# Introduce to neural network
greetUser()

# Train our neural network
network = NeuralNetwork()
network.train(data, all_y_trues)
print("Network training has been completed.")

#w = np.array([-35, -2]) # 128 pounds, 63 inches
#m = np.array([20, 2])  # 155 pounds, 68 inches

test = np.array([-15, 7])
print(f"test: {network.feedforward(test)}")

#if (1 - network.feedforward(test)) < 0.5:
#    print(f"Person: {network.feedforward(test)}")

# User Interaction Loop
while bolTrueFalse:
    # Step 1. Ask for primer confirmation
    userConfirmation = str(input("\nWould you like to run a prediction? (Y/N): "))
    try:
        if userConfirmation == "Y":
            # Step 2. Would they prefer to enter their own data points?
            userTrainChoice = int(input("\nWhat data would you like to enter?\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n(1) Your own specifications\n(2) Random datasets\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\nPlease enter 1 or 2 to choose:"))
            while True:
                if userTrainChoice == 1:
                    # Step 3. Ask the user for specified data
                    try:
                        userName = input("\nWhat is your name?: ")
                        userWeight = float(input("\nEnter your weight in pounds: ")) - 135
                        userHeight = float(input("\nEnter your height in inches: ")) - 66
                        userData = np.array([userWeight, userHeight])
                        print("Predicting gender in...\n3")
                        t.sleep(1)
                        print("2")
                        t.sleep(1)
                        print("1")
                        t.sleep(1)
                        if (network.feedforward(userData)) < 0.1:
                            print(f"{userName} is extremely likely to be male.\nExact expectation: {network.feedforward(userData)} ")
                        elif (network.feedforward(userData)) < 0.5:
                            print(f"{userName} is likely a male.\nExact expectation: {network.feedforward(userData)} ")
                        elif (network.feedforward(userData)) > 0.9:
                            print(f"{userName} is extremely likely to be female.\nExact expectation: {network.feedforward(userData)} ")
                        elif (network.feedforward(userData)) > 0.5:
                            print(f"{userName} is likely a female.\nExact expectation: {network.feedforward(userData)} ")
                        else:
                            print(f"Unreachable error reached.")
                        break
                    except:
                        # 
                        print("Invalid specifications - please try again.")
                        continue



                    break
                elif userTrainChoice == 2:
                    

                    break
                else:
                    print("Did not enter 1 or 2 - please try again.")
                    continue
        elif userConfirmation == "N":
            print("\nHave a good day.\n\n\n\n\n░░░░░░░░░░░░░░░░░ NETWORK ENDED ░░░░░░░░░░░░░░░░░")
            break
        else:
            print("\nDid not enter Y or N - please try again.")
    except:
        print("\nInvalid input - please try again")
