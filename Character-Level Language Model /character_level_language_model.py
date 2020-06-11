#!/usr/bin/env python
# coding: utf-8

# # Character level language model - Dinosaurus Island
# 
# Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and in this model they are back. We are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, and your job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go berserk, so choose wisely! 
# 
# <table>
# <td>
# <img src="images/dino.jpg" style="width:250;height:300px;">
# 
# </td>
# 
# </table>
# 
# Luckily we have learned some deep learning and we will use it to save the day. We have collected a list of all the dinosaur names we could find, and compiled them into this [dataset](dinos.txt). To create new dinosaur names, we will build a character level language model to generate new names. Our algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep us and our team safe from the dinosaurs' wrath! 
# 
# We will begin by loading in some functions that we have in `rnn_utils`. Specifically, we have access to functions such as `rnn_forward` and `rnn_backward` which are equivalent to those we've implemented in the previous models. 

# In[1]:


import numpy as np
from utils import *
import random
import pprint


# ### 1.1 - Dataset and Preprocessing
# 
# Run the following cell to read the dataset of dinosaur names, create a list of unique characters (such as a-z), and compute the dataset and vocabulary size. 

# In[2]:


data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


# * `char_to_ix`: In the cell below, we create a python dictionary (i.e., a hash table) to map each character to an index from 0-26.
# * `ix_to_char`: We also create a second python dictionary that maps each index back to the corresponding character. 
#     -  This will help us figure out what index corresponds to what character in the probability distribution output of the softmax layer. 

# In[3]:


chars = sorted(chars)
print(chars)


# In[4]:


char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)


# ### 1.2 - Overview of the model
# 
# Our model will have the following structure: 
# 
# - Initialize parameters 
# - Run the optimization loop
#     - Forward propagation to compute the loss function
#     - Backward propagation to compute the gradients with respect to the loss function
#     - Clip the gradients to avoid exploding gradients
#     - Using the gradients, update your parameters with the gradient descent update rule.
# - Return the learned parameters 
#     
# <img src="images/rnn.png" style="width:450;height:300px;">
# <caption><center> **Figure 1**: Recurrent Neural Network, similar to what you had built in the previous notebook "Building a Recurrent Neural Network - Step by Step".  </center></caption>
# 
# * At each time-step, the RNN tries to predict what is the next character given the previous characters. 
# * The dataset $\mathbf{X} = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle})$ is a list of characters in the training set.
# * $\mathbf{Y} = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_x \rangle})$ is the same list of characters but shifted one character forward. 
# * At every time-step $t$, $y^{\langle t \rangle} = x^{\langle t+1 \rangle}$.  The prediction at time $t$ is the same as the input at time $t + 1$.

# ## 2 - Building blocks of the model
# 
# In this part, we will build two important blocks of the overall model:
# - Gradient clipping: to avoid exploding gradients
# - Sampling: a technique used to generate characters
# 
# We will then apply these two functions to build the model.

# ### 2.1 - Clipping the gradients in the optimization loop
# 
# In this section we will implement the `clip` function that we will call inside of your optimization loop. 
# 
# #### Exploding gradients
# * When gradients are very large, they're called "exploding gradients."  
# * Exploding gradients make the training process more difficult, because the updates may be so large that they "overshoot" the optimal values during back propagation.
# 
# Recall that our overall loop structure usually consists of:
# * forward pass, 
# * cost computation, 
# * backward pass, 
# * parameter update. 
# 
# Before updating the parameters, we will perform gradient clipping to make sure that your gradients are not "exploding."
# 
# #### gradient clipping
# Let's implement a function `clip` that takes in a dictionary of gradients and returns a clipped version of gradients if needed.  
# * For example, if the N=10
#     - The range is [-10, 10]
#     - If any component of the gradient vector is greater than 10, it is set to 10.
#     - If any component of the gradient vector is less than -10, it is set to -10. 
#     - If any components are between -10 and 10, they keep their original values.
# 
# <img src="images/clip.png" style="width:400;height:150px;">
# <caption><center> **Figure 2**: Visualization of gradient descent with and without gradient clipping, in a case where the network is running into "exploding gradient" problems. </center></caption>

# In[9]:


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients


# In[10]:


# Test with a maxvalue of 10
maxValue = 10
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, maxValue)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])


# In[11]:


# Test with a maxValue of 5
maxValue = 5
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, maxValue)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])


# ### 2.2 - Sampling
# 
# Now assume that our model is trained. We would like to generate new text (characters). The process of generation is explained in the picture below:
# 
# <img src="images/dinos3.png" style="width:500;height:300px;">
# <caption><center> **Figure 3**: In this picture, we assume the model is already trained. We pass in $x^{\langle 1\rangle} = \vec{0}$ at the first time step, and have the network sample one character at a time. </center></caption>

# Let's implement the `sample` function below to sample characters. We need to carry out 4 steps:
# 
# - **Step 1**: Input the "dummy" vector of zeros $x^{\langle 1 \rangle} = \vec{0}$. 
#     - This is the default input before we've generated any characters. 
#     We also set $a^{\langle 0 \rangle} = \vec{0}$

# - **Step 2**: Run one step of forward propagation to get $a^{\langle 1 \rangle}$ and $\hat{y}^{\langle 1 \rangle}$. Here are the equations:
# 
# hidden state:  
# $$ a^{\langle t+1 \rangle} = \tanh(W_{ax}  x^{\langle t+1 \rangle } + W_{aa} a^{\langle t \rangle } + b)\tag{1}$$
# 
# activation:
# $$ z^{\langle t + 1 \rangle } = W_{ya}  a^{\langle t + 1 \rangle } + b_y \tag{2}$$
# 
# prediction:
# $$ \hat{y}^{\langle t+1 \rangle } = softmax(z^{\langle t + 1 \rangle })\tag{3}$$

# In[12]:


import numpy as np


# In[13]:


matrix1 = np.array([[1,1],[2,2],[3,3]]) # (3,2)
matrix2 = np.array([[0],[0],[0]]) # (3,1) 
vector1D = np.array([1,1]) # (2,) 
vector2D = np.array([[1],[1]]) # (2,1)
print("matrix1 \n", matrix1,"\n")
print("matrix2 \n", matrix2,"\n")
print("vector1D \n", vector1D,"\n")
print("vector2D \n", vector2D)


# In[14]:


print("Multiply 2D and 1D arrays: result is a 1D array\n", 
      np.dot(matrix1,vector1D))
print("Multiply 2D and 2D arrays: result is a 2D array\n", 
      np.dot(matrix1,vector2D))


# In[15]:


print("Adding (3 x 1) vector to a (3 x 1) vector is a (3 x 1) vector\n",
      "This is what we want here!\n", 
      np.dot(matrix1,vector2D) + matrix2)


# In[16]:


print("Adding a (3,) vector to a (3 x 1) vector\n",
      "broadcasts the 1D array across the second dimension\n",
      "Not what we want here!\n",
      np.dot(matrix1,vector1D) + matrix2
     )


# - **Step 3**: Sampling: 
#     - Now that we have $y^{\langle t+1 \rangle}$, we want to select the next letter in the dinosaur name. If we select the most probable, the model will always generate the same result given a starting letter. 
#         - To make the results more interesting, we will use np.random.choice to select a next letter that is likely, but not always the same.
#     - Sampling is the selection of a value from a group of values, where each value has a probability of being picked.  
#     - Sampling allows us to generate random sequences of values.
#     - Pick the next character's index according to the probability distribution specified by $\hat{y}^{\langle t+1 \rangle }$. 
#     - This means that if $\hat{y}^{\langle t+1 \rangle }_i = 0.16$, you will pick the index "i" with 16% probability. 
#     - This means that you will pick the index (`idx`) according to the distribution: 
# 
#     $P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$.
# 
#     - Note that the value that's set to `p` should be set to a 1D vector.
#     - Also notice that $\hat{y}^{\langle t+1 \rangle}$, which is `y` in the code, is a 2D array.

# - **Step 4**: Update to $x^{\langle t \rangle }$ 
#     - The last step to implement in `sample()` is to update the variable `x`, which currently stores $x^{\langle t \rangle }$, with the value of $x^{\langle t + 1 \rangle }$. 
#     - We will represent $x^{\langle t + 1 \rangle }$ by creating a one-hot vector corresponding to the character that we have chosen as your prediction. 
#     - We will then forward propagate $x^{\langle t + 1 \rangle }$ in Step 1 and keep repeating the process until we get a "\n" character, indicating that we have reached the end of the dinosaur name. 

# In[17]:


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # Step 1: Create the a zero vector x that can be used as the one-hot vector 
    # representing the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    
    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # We will initialize idx to -1
    idx = -1 
    
    # Loop over time-steps t. At each time-step:
    # sample a character from a probability distribution 
    # and append its index (`idx`) to the list "indices". 
    # We'll stop if we reach 50 characters 
    # (which should be very unlikely with a well trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        # for grading purposes
        np.random.seed(counter+seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        # for grading purposes
        seed += 1
        counter +=1
        

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices


# In[18]:


np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:\n", indices)
print("list of sampled characters:\n", [ix_to_char[i] for i in indices])


# ## 3 - Building the language model 
# 
# It is time to build the character-level language model for text generation. 
# 
# 
# ### 3.1 - Gradient descent 
# 
# * In this section you will implement a function performing one step of stochastic gradient descent (with clipped gradients). 
# * We will go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent. 
# 
# As a reminder, here are the steps of a common optimization loop for an RNN:
# 
# - Forward propagate through the RNN to compute the loss
# - Backward propagate through time to compute the gradients of the loss with respect to the parameters
# - Clip the gradients
# - Update the parameters using gradient descent 
# 
# Let's implement the optimization process (one step of stochastic gradient descent). 

# #### parameters
# 
# * Note that the weights and biases inside the `parameters` dictionary are being updated by the optimization, even though `parameters` is not one of the returned values of the `optimize` function. The `parameters` dictionary is passed by reference into the function, so changes to this dictionary are making changes to the `parameters` dictionary even when accessed outside of the function.

# In[19]:


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1]


# In[20]:


np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])


# ### 3.2 - Training the model 

# * Given the dataset of dinosaur names, we use each line of the dataset (one name) as one training example. 
# * Every 100 steps of stochastic gradient descent, you will sample 10 randomly chosen names to see how the algorithm is doing. 
# * Remember to shuffle the dataset, so that stochastic gradient descent visits the examples in random order. 
# 
# Let's implement `model()`. When `examples[index]` contains one dinosaur name (string), to create an example (X, Y), you can use this:
# 
# ##### Set the index `idx` into the list of examples
# 
# ##### Extract a single example from the list of examples
# * `single_example`: use the `idx` index that we set previously to get one word from the list of examples.

# ##### Convert a string into a list of characters: `single_example_chars`
# * `single_example_chars`: A string is a list of characters.
# * You can use a list comprehension (recommended over for-loops) to generate a list of characters.

# ##### Convert list of characters to a list of integers: `single_example_ix`
# * Create a list that contains the index numbers associated with each character.
# * Use the dictionary `char_to_ix`
# * We can combine this with the list comprehension that is used to get a list of characters from a string.

# ##### Create the list of input characters: `X`
# * `rnn_forward` uses the `None` value as a flag to set the input vector as a zero-vector.
# * Prepend the `None` value in front of the list of input characters.

# ##### Get the integer representation of the newline character `ix_newline`
# * `ix_newline`: The newline character signals the end of the dinosaur name.
#     - get the integer representation of the newline character `'\n'`.
#     - Use `char_to_ix`

# ##### Set the list of labels (integer representation of the characters): `Y`
# * The goal is to train the RNN to predict the next letter in the name, so the labels are the list of characters that are one time step ahead of the characters in the input `X`.
#     - For example, `Y[0]` contains the same value as `X[1]`  
# * The RNN should predict a newline at the last letter so add ix_newline to the end of the labels. 
#     - Append the integer representation of the newline character to the end of `Y`.

# In[28]:


def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(num_iterations):
        
        
        # Set the index `idx` (see instructions above)
        idx = j % len(examples)
        
        # Set the input X (see instructions above)
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        single_example_ix = [char_to_ix[c]for c in single_example_chars]
        X = [None] + single_example_ix
        
        # Set the labels Y (see instructions above)
        ix_newline = char_to_ix["\n"]
        Y = X[1:] + [ix_newline]
        
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
      
            print('\n')
        
    return parameters


# Run the following cell, you should observe your model outputting random-looking characters at the first iteration. 

# In[29]:


parameters = model(data, ix_to_char, char_to_ix)


# ## Conclusion
# 
# We can see that our algorithm has started to generate plausible dinosaur names towards the end of the training. At first, it was generating random characters, but towards the end we could see dinosaur names with cool endings. Our implementation generated some really cool names like `maconucon`, `marloralus` and `macingsersaurus`. Our model hopefully also learned that dinosaur names tend to end in `saurus`, `don`, `aura`, `tor`, etc.
# 
# <img src="images/mangosaurus.jpeg" style="width:250;height:300px;">

# ## 4 - Writing like Shakespeare
# 
# A similar (but more complicated) task is to generate Shakespeare poems. Instead of learning from a dataset of Dinosaur names we can use a collection of Shakespearian poems. Using LSTM cells, we can learn longer term dependencies that span many characters in the text--e.g., where a character appearing somewhere a sequence can influence what should be a different character much much later in the sequence. These long term dependencies were less important with dinosaur names, since the names were quite short. 
# 
# 
# <img src="images/shakespeare.jpg" style="width:500;height:400px;">
# <caption><center> Let's become poets! </center></caption>
# 
# We have implemented a Shakespeare poem generator with Keras. Run the following cell to load the required packages and models.

# In[30]:


from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io


# To save some time, we have already trained a model for ~1000 epochs on a collection of Shakespearian poems called [*"The Sonnets"*](shakespeare.txt). 

# Let's train the model for one more epoch. When it finishes training for an epoch---this will also take a few minutes---you can run `generate_output`, which will prompt asking you for an input (`<`40 characters). The poem will start with your sentence, and our RNN-Shakespeare will complete the rest of the poem for you! For example, try "Forsooth this maketh no sense " (don't enter the quotation marks). Depending on whether you include the space at the end, your results might also differ--try it both ways, and try other inputs as well. 
# 

# In[31]:


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])


# In[32]:


# Run this cell to try with different inputs without having to re-train the model 
generate_output()


# The RNN-Shakespeare model is very similar to the one you have built for dinosaur names. The only major differences are:
# - LSTMs instead of the basic RNN to capture longer-range dependencies
# - The model is a deeper, stacked LSTM model (2 layer)
# - Using Keras instead of python to simplify the code 

# **References**:
# - This exercise took inspiration from Andrej Karpathy's implementation: https://gist.github.com/karpathy/d4dee566867f8291f086. To learn more about text generation, also check out Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
# - For the Shakespearian poem generator, our implementation was based on the implementation of an LSTM text generator by the Keras team: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 
