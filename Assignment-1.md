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


### Data :
First, we have to prepare the `iris` data,


```
from sklearn import datasets

iris = datasets.load_iris()

iris.data[:5]

array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2]])
```

Now, let's check the label of the data 

```
iris.target

## Output

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

```

This is a long list of a vector with 3 categories. We have to convert it using `one-hot encoding`

```

one_hot_encoder = OneHotEncoder(sparse=False)

Y = iris.target
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

Y[:5]

## Output
array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.]])

```

Now, Let's do the test train split,

```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)
```



### Implementation 


#### NeuralNetwork Function :

```

def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=100, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if(epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(Accuracy(X_train, Y_train, 
            weights)))
            if X_val.any():
                print("Validation Accuracy:{}".format(Accuracy(X_val, Y_val, 
                weights)))
             
    return weights
```


#### Initializing weights :

The wieghts are randomly initialized from -1 to 1.
```
def InitializeWeights(nodes):
    layers, weights = len(nodes), []
    
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
              for j in range(nodes[i])]
        weights.append(np.matrix(w))
    
    return weights
```


#### Forward Propagation and Backward Propagation :

The training of the weights is accomplished with forward and backprop function.


```
def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    
    return activations





def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights
```



### Train Function :
We are passing each sample of our dataset through the network with forward propagation and then the weights are updated using the backpropation function.
Finally, the updated weights will be returned.


```
def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        
        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights
```


#### Activation Function :

```
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)
```


#### Predict function :

The output of the network will be in a `[x, y, z]` vector form with a number range of `[0,1]`. The highest value of the vector is highly likely to be the correct calss.

```
def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y # Return prediction vector

def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    
    return index
```

#### Accuracy function :

```
def Accuracy(X, Y, weights):
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if(y == guess):
            # Guessed correctly
            correct += 1

    return correct / len(X)
```


#### Output :

Now we run the network with  training and validation set, epochs, learning rate, and number of nodes for every layer.



```


# Input : 4 nodes 
# Hidden : 8 nodes 
# Output : 3 nodes

layers = [4, 8, 3] # Number of nodes in layers
lr, epochs = 0.15, 100

weights = NeuralNetwork(X_train, Y_train, X_val, Y_val, epochs=epochs, nodes=layers, lr=lr



### Output 

Epoch 80
Training Accuracy:0.9537037037037037
Validation Accuracy:1.0
Epoch 100
Training Accuracy:0.9537037037037037
Validation Accuracy:1.0

## Test accuracy :
Accuracy(X_test, Y_test, weights)
Testing Accuracy: 0.9652173913034348

```
##### Reference : 
[Kaggle  Notebook](https://www.kaggle.com/code/antmarakis/another-neural-network-from-scratch/notebook)





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



We followed The skip-gram version of the Tensorflow tutorial. All the insctruction can be found [here](https://www.tensorflow.org/tutorials/text/word2vec#skip-gram_and_negative_sampling.)


I am only showing some snippet of the imporant parts   



#### Loading the data :

After downloading the american standard version of the bible `t_asv`, we load the data


```
with open("/content/t_asv.csv") as f:
  lines = f.read().splitlines()
for line in lines[:20]:
  print(line)


### Output
id,b,c,v,t
1001001,1,1,1,In the beginning God created the heavens and the earth.
1001002,1,1,2,And the earth was waste and void; and darkness was upon the face of the deep: and the Spirit of God moved upon the face of the waters.
```


### Setup :

#### Let's create a custom standardization function to lowercase the text and remove punctuation.

```

def custom_standardization(input_data):

lowercase = tf.strings.lower(input_data)

return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation),'')
```


#### Define the vocabulary size and the number of words in a sequence :
```
vocab_size = 4096

sequence_length = 10
```

#### Slecting batch and buffer :

```
BATCH_SIZE = 1024

BUFFER_SIZE = 10000
```

#### Selecting optimizer and loss function :

```
embedding_dim = 128

word2vec = Word2Vec(vocab_size, embedding_dim)

word2vec.compile(optimizer='adam',

loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

metrics=['accuracy'])
```


#### Training ther model with 20 epochs :

```
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
```

Now that we have trained our model we have the trained weights which are the vectors of the words. And, we can make `vectors.tsv` and `metdata.tsv` file and load them up on `porjector.tensorflow`. 


![[Screenshot 2022-05-08 at 5.38.57 PM.png]]





### Nearest Neighbours :

To find the NN we will be using consine distance function 

```
def cosine_sim(a, b):
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
```

Now we  put each word with their related vectors into a dictionary.

```
word2vec_dict = {vocab[i]: weights[i] for i in range(len(vocab))}

```

Then we need a function that can measure the top 10 Nearest neighbour.

```
def similarity(a, k=10):
	vector = word2vec_dict[a]
	
	similar_word = {}
	
	for key in word2vec_dict:
		distance = cosine_sim(vector, word2vec_dict[key])
		similar_word[key] = distance
		
	sorted_word = sorted(similar_word.items(), key=lambda kv: kv[1])
	sorted_word.reverse()
	sorted_word = sorted_word[1:k+1]
	return sorted_word
```



#### Now we can Output the 10 nearest neighbors for the terms


1) holy
```

similarity('holy')


[('convocation', 0.4765327),
 ('kingdoms', 0.4493326),
 ('satisfied', 0.4461348),
 ('terrors', 0.44165617),
 ('offspring', 0.44125158),
 ('dance', 0.43971455),
 ('precious', 0.43510044),
 ('standard', 0.42500424),
 ('censer', 0.4243147),
 ('fellowship', 0.42137915)]
```



2) woman

```
similarity('woman')

[('thief', 0.5304504),
 ('overcome', 0.5241327),
 ('leaven', 0.5188071),
 ('resurrection', 0.5184521),
 ('bright', 0.50045574),
 ('elder', 0.49653435),
 ('oft', 0.4818967),
 ('tribute', 0.47746423),
 ('hushai', 0.47220024),
 ('bullock', 0.4697402)]

```
3) light 

```
similarity('light')

[('cleansed', 0.47259983),
 ('thirsty', 0.453699),
 ('bird', 0.45217338),
 ('worship', 0.4484667),
 ('certainly', 0.4435583),
 ('gentle', 0.44352266),
 ('contention', 0.42491394),
 ('ashdod', 0.42310148),
 ('forgotten', 0.41951954),
 ('praying', 0.40924448)]
```





####  Output the top-3 nearest neighbors for the algebraic expression :



We already built a dictionary, now we just have to search the word's respective vector  and then we can `Add` or `substract` them.

We have a  function for measuring similarity. We can use that one or  another similar function called `algebraic_similarity` that only accepts vector as the parameter.


1) jesus-man+woman=

```
first_word_vector = np.subtract(np.add(word2vec_dict["man"],word2vec_dict["woman"]),word2vec_dict['jesus'])

# Running the function
algebraic_similarity(first_word_vector)


# Output 
[('woman', 0.6120108), ('whereupon', 0.497809), ('person', 0.47751632), 
```

2) money-evil+good=

```
second_word_vector = 
np.subtract(np.add(word2vec_dict["evil"],word2vec_dict["good"]),word2vec_dict['money'])


algebraic_similarity(second_word_vector)


### Output
[('good', 0.7648879), ('slow', 0.4411451), ('dieth', 0.4377962), ('perverse', 0.43004104)]

```


3) find an interesting expression on your own
```

np.add(word2vec_dict["jesus"],word2vec_dict["heaven"])


### Output
[('jesus', 0.72261715),
 ('woman', 0.5546671),
 ('wickedness', 0.4895949),
 ('nation', 0.48249346),

```