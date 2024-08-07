from scipy.io import loadmat 
import numpy as np 
from scipy.optimize import minimize 
import cv2

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

# Load the image using OpenCV
image_path = "C:\\Users\\vinay\\Downloads\\opencv-course\\opencv-course\\saved_image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))

# Extracting pixel matrix of image and converting it to a vector of (1, 784) 
x = np.asarray(image)
 
vec = np.zeros((1, 784)) 
k = 0
for i in range(28): 
    for j in range(28): 
        vec[0][k] = x[i][j] 
        k += 1
# Loading Thetas 
Theta1 = np.loadtxt('Theta1.txt') 
Theta2 = np.loadtxt('Theta2.txt') 
  
# Calling function for prediction 
pred = predict(Theta1, Theta2, 1 - vec / 255) 
print(pred[0])
    

