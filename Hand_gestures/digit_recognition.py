from scipy.io import loadmat 
import numpy as np 
from scipy.optimize import minimize 
import cv2

def initialise(a, b): 
	epsilon = 0.15
	c = np.random.rand(a, b + 1) * ( 
	# Randomly initialises values of thetas between [-epsilon, +epsilon] 
	2 * epsilon) - epsilon 
	return c 

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb): 
    # Weights are split back to Theta1, Theta2 
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], 
                        (hidden_layer_size, input_layer_size + 1)) 
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],  
                        (num_labels, hidden_layer_size + 1)) 
  
    # Forward propagation 
    m = X.shape[0] 
    one_matrix = np.ones((m, 1)) 
    X = np.append(one_matrix, X, axis=1)  # Adding bias unit to first layer 
    a1 = X 
    z2 = np.dot(X, Theta1.transpose()) 
    a2 = 1 / (1 + np.exp(-z2))  # Activation for second layer 
    one_matrix = np.ones((m, 1)) 
    a2 = np.append(one_matrix, a2, axis=1)  # Adding bias unit to hidden layer 
    z3 = np.dot(a2, Theta2.transpose()) 
    a3 = 1 / (1 + np.exp(-z3))  # Activation for third layer 
  
    # Changing the y labels into vectors of boolean values. 
    # For each label between 0 and 9, there will be a vector of length 10 
    # where the ith element will be 1 if the label equals i 
    y_vect = np.zeros((m, 10)) 
    for i in range(m): 
        y_vect[i, int(y[i])] = 1
  
    # Calculating cost function 
    J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (lamb / (2 * m)) * ( 
                sum(sum(pow(Theta1[:, 1:], 2))) + sum(sum(pow(Theta2[:, 1:], 2)))) 
  
    # backprop 
    Delta3 = a3 - y_vect 
    Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2) 
    Delta2 = Delta2[:, 1:] 
  
    # gradient 
    Theta1[:, 0] = 0
    Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (lamb / m) * Theta1 
    Theta2[:, 0] = 0
    Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (lamb / m) * Theta2 
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten())) 
  
    return J, grad 

def predict(Theta1, Theta2, X): 
    m = X.shape[0] 
    one_matrix = np.ones((m, 1)) 
    X = np.append(one_matrix, X, axis=1)  # Adding bias unit to first layer 
    z2 = np.dot(X, Theta1.transpose()) 
    a2 = 1 / (1 + np.exp(-z2))  # Activation for second layer 
    one_matrix = np.ones((m, 1)) 
    a2 = np.append(one_matrix, a2, axis=1)  # Adding bias unit to hidden layer 
    z3 = np.dot(a2, Theta2.transpose()) 
    a3 = 1 / (1 + np.exp(-z3))  # Activation for third layer 
    p = (np.argmax(a3, axis=1))  # Predicting the class on the basis of max value of hypothesis 
    return p 

# Loading mat file 
data = loadmat('mnist-original.mat') 

# Extracting features from mat file 
X = data['data'] 
X = X.transpose() 

# Normalizing the data 
X = X / 255

# Extracting labels from mat file 
y = data['label'] 
y = y.flatten() 

# Splitting data into training set with 60,000 examples 
X_train = X[:60000, :] 
y_train = y[:60000] 

# Splitting data into testing set with 10,000 examples 
X_test = X[60000:, :] 
y_test = y[60000:] 

m = X.shape[0] 
input_layer_size = 784 # Images are of (28 X 28) px so there will be 784 features 
hidden_layer_size = 100
num_labels = 10 # There are 10 classes [0, 9] 

# Randomly initialising Thetas 
initial_Theta1 = initialise(hidden_layer_size, input_layer_size) 
initial_Theta2 = initialise(num_labels, hidden_layer_size) 

# Unrolling parameters into a single column vector 
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten())) 
maxiter = 100
lambda_reg = 0.1 # To avoid overfitting 
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg) 

# Calling minimize function to minimize cost function and to train weights 
results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
		options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True) 

nn_params = results["x"] # Trained Theta is extracted 

# Weights are split back to Theta1, Theta2 
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], ( 
							hidden_layer_size, input_layer_size + 1)) # shape = (100, 785) 
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
					(num_labels, hidden_layer_size + 1)) # shape = (10, 101) 

# Checking test set accuracy of our model 
pred = predict(Theta1, Theta2, X_test) 
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100))) 

# Checking train set accuracy of our model 
pred = predict(Theta1, Theta2, X_train) 
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100))) 

# Evaluating precision of our model 
true_positive = 0
for i in range(len(pred)): 
	if pred[i] == y_train[i]: 
		true_positive += 1
false_positive = len(y_train) - true_positive 
print('Precision =', true_positive/(true_positive + false_positive)) 

# Saving Thetas in .txt file 
np.savetxt('Theta1.txt', Theta1, delimiter=' ') 
np.savetxt('Theta2.txt', Theta2, delimiter=' ') 