## Task 1 : Liner Algebra
 
 $X = \begin{bmatrix}  2 & 4 \\ 1 & 3  \end{bmatrix}$

 $y = \begin{bmatrix}  1 \\ 3  \end{bmatrix}$

$z = \begin{bmatrix}  2 \\ 3  \end{bmatrix}$

  
  
a) $y^T z$ = $11$

b) $Xy = \begin{bmatrix}  14 \\ 10  \end{bmatrix}$
  
c) $X^2 = \begin{bmatrix}  8 & 20 \\ 5 & 13  \end{bmatrix}$

d) $X^-1$ : To find the inverse of a matrix, we augment the matrix with the identity matrix and solve the system of equations.

And this is what we find:

 $X^-1 = \begin{bmatrix}  1.5 & -2 \\ -0.5 & 1  \end{bmatrix}$

e) $Rank(X)$ = The rank of $X$ is 2.



e) rank(X) = The rank of X is 2.

---

## Task 2 : Derivatives

  
Derivative of $y$ with respect to $x$ , if
  

a) $y = x^3 + x - 5 = 3x^2 + 1$

b) $y = (5x^3 - 2x) (2x) = 40*x^3-8x$ 

c) $\dfrac{2x^2+3}{8x+1} = \dfrac{4\left(4x^2+x-6\right)}{\left(8x+1\right)^2}$  
  
d) $(3*x-2)^8$ = $24*(3x-2)^7$

e) $(logx^2 + x) = ln(x)^2+x = \dfrac{2\ln\left(x\right)}{x}+1$


## Task 3 : Cross Entropy Gradient

Derive the gradient of the cross entropy cost function with respect to the input of the softmax function.

- In practice, the softmax function is usually combined with cross entropy loss.

---

## Task 4 : Neural Network Gradient

a) Derive the gradient of cross entropy loss with respect to the input of the neural network.

b) how many parameters does the neural network have?
---

## Task 5 : Implementing Backpropagation

Implement a neural network and the backparopagation algorithm for the iris dataset. You can load it in python, e.g., through from sklearn.datasets import load iris. Explore the data to understand the features and the classes, e.g., by looking at the data distributions.
Don’t use any libraries for neural networks (scikit.learn, keras, etc.) but only python, numpy, pandas, etc. You can hard-code the network architecture:
• 4 input nodes
• 1 hidden layer with 8 nodes and relu activation function • 3 output nodes with softmax activation function
• cross entropy as loss function
Split the data into 80% training and 20% test with balanced class distributions Initialize your weights randomly and then train your network for 100 epochs.
Try this without help from the Web first. If you get stuck, you can consult the Web: there are many tutorials on using the iris dataset, e.g., https://janakiev.com/blog/keras-iris/ and on implementing backprop, e.g., https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/. 30 P
a) Output the accuracy of your trained model on the training data.
b) Output the accuracy of your trained model on the test data. 

---

## Task 6 : Implementing word2vec

Implement your own word2vec method to create word embeddings. For training, use the amer-ican standard version (asv) bible. You can find it, e.g., here: https://www.kaggle.com/dat asets/oswinrh/bible. You can follow the example at https://www.tensorflow.org/tutorials/text/word2vec. They implement the skip-gram version of word2vec. Optionally, you could also implement the CBOW version. In any case, store your learned embeddings in a format to be able to load it into http://projector.tensorflow.org/ for visualization.

a) Output the 10 nearest neighbors for the term 1) holy
2) woman
3) light
b) Output the top-3 nearest neighbors for the algebraic expression
1) jesus-man+woman=?
2) money-evil+good=?
3) find an interesting expression on your own