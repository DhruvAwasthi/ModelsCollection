#!/usr/bin/env python
# coding: utf-8

# # Improvise a Jazz Solo with an LSTM Network
# In this model, we will
# - Apply an LSTM to music generation.
# - Generate our own jazz music with deep learning.

# Please run the following cell to load all the packages required in this assignment. This may take a few minutes. 

# In[1]:


from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


# ## 1 - Understanding the Problem statement
# 
# You would like to create a jazz music piece specially for a friend's birthday. However, you don't know any instruments or music composition. Fortunately, you know deep learning and will solve this problem using an LSTM network.  
# 
# You will train a network to generate novel jazz solos in a style representative of a body of performed work.
# 
# <img src="images/jazz.jpg" style="width:450;height:300px;">
# 
# 
# ### 1.1 - Dataset
# 
# You will train your algorithm on a corpus of Jazz music. Run the cell below to listen to a snippet of the audio from the training set:

# In[2]:


IPython.display.Audio('./data/30s_seq.mp3')


# We have taken care of the preprocessing of the musical data to render it in terms of musical "values." 
# 
# Run the following code to load the raw music data and preprocess it into values.

# In[3]:


X, Y, n_values, indices_values = load_music_utils()
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('shape of X:', X.shape)
print('Shape of Y:', Y.shape)


# We have just loaded the following:
# 
# - `X`: This is an (m, $T_x$, 78) dimensional array. 
#     - We have m training examples, each of which is a snippet of $T_x =30$ musical values. 
#     - At each time step, the input is one of 78 different possible values, represented as a one-hot vector. 
#         - For example, X[i,t,:] is a one-hot vector representing the value of the i-th example at time t. 
# 
# - `Y`: a $(T_y, m, 78)$ dimensional array
#     - This is essentially the same as `X`, but shifted one step to the left (to the past). 
#     - Notice that the data in `Y` is **reordered** to be dimension $(T_y, m, 78)$, where $T_y = T_x$. This format makes it more convenient to feed into the LSTM later.
#     - We're using the previous values to predict the next value.
#         - So our sequence model will try to predict $y^{\langle t \rangle}$ given $x^{\langle 1\rangle}, \ldots, x^{\langle t \rangle}$. 
# 
# - `n_values`: The number of unique values in this dataset. This should be 78. 
# 
# - `indices_values`: python dictionary mapping integers 0 through 77 to musical values.

# ### 1.2 - Overview of our model
# 
# Here is the architecture of the model we will use. This is similar to the Dinosaurus model, except that you will implement it in Keras.
# 
# <img src="images/music_generation.png" style="width:600;height:400px;">
# 
# 
# * $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \cdots, x^{\langle T_x \rangle})$ is a window of size $T_x$ scanned over the musical corpus. 
# * Each $x^{\langle t \rangle}$ is an index corresponding to a value.
# * $\hat{y}^{t}$ is the prediction for the next value.
# * We will be training the model on random snippets of 30 values taken from a much longer piece of music. 
#     - Thus, we won't bother to set the first input $x^{\langle 1 \rangle} = \vec{0}$, since most of these snippets of audio start somewhere in the middle of a piece of music. 
#     - We are setting each of the snippets to have the same length $T_x = 30$ to make vectorization easier.

# ## Overview of parts 2 and 3
# 
# * We're going to train a model that predicts the next note in a style that is similar to the jazz music that it's trained on.  The training is contained in the weights and biases of the model. 
# * In Part 3, we're then going to use those weights and biases in a new model which predicts a series of notes, using the previous note to predict the next note. 
# * The weights and biases are transferred to the new model using 'global shared layers' described below"
# 

# ## 2 - Building the model
# 
# * In this part we will build and train a model that will learn musical patterns. 
# * The model takes input X of shape $(m, T_x, 78)$ and labels Y of shape $(T_y, m, 78)$. 
# * We will use an LSTM with hidden states that have $n_{a} = 64$ dimensions.

# In[4]:


# number of dimensions for the hidden state of each LSTM cell.
n_a = 64 


# 
# #### Sequence generation uses a for-loop
# * If we're building an RNN where, at test time, the entire input sequence $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \ldots, x^{\langle T_x \rangle}$ is given in advance, then Keras has simple built-in functions to build the model. 
# * However, for **sequence generation, at test time we don't know all the values of $x^{\langle t\rangle}$ in advance**.
# * Instead we generate them one at a time using $x^{\langle t\rangle} = y^{\langle t-1 \rangle}$. 
#     * The input at time "t" is the prediction at the previous time step "t-1".
# * So we'll need to implement your own for-loop to iterate over the time steps. 
# 
# #### Shareable weights
# * The function `djmodel()` will call the LSTM layer $T_x$ times using a for-loop.
# * It is important that all $T_x$ copies have the same weights. 
#     - The $T_x$ steps should have shared weights that aren't re-initialized.
# * Referencing a globally defined shared layer will utilize the same layer-object instance at each time step.
# * The key steps for implementing layers with shareable weights in Keras are: 
# 1. Define the layer objects (we will use global variables for this).
# 2. Call these objects when propagating the input.

# In[5]:


n_values = 78 # number of music values
reshapor = Reshape((1, n_values))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D


# Let's implement `djmodel()`. 
# 
# #### Inputs (given)
# * The `Input()` layer is used for defining the input `X` as well as the initial hidden state 'a0' and cell state `c0`.
# * The `shape` parameter takes a tuple that does not include the batch dimension (`m`).
# 
# #### Step 1: Outputs (TODO)
# 1. Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.

# #### Step 2: Loop through time steps (TODO)
# * Loop for $t \in 1, \ldots, T_x$:
# 
# #### 2A. Select the 't' time-step vector from X.

# #### 2B. Reshape x to be (1,n_values).
# 
# #### 2C. Run x through one step of LSTM_cell.
# 
# #### 2D. Dense layer
# * Propagate the LSTM's hidden state through a dense+softmax layer using `densor`. 
#     
# #### 2E. Append output
# * Append the output to the list of "outputs".
# 

# #### Step 3: After the loop, create the model

# In[12]:


def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
    
    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model


# #### Create the model object
# * Run the following cell to define our model. 

# In[13]:


model = djmodel(Tx = 30 , n_a = 64, n_values = 78)


# In[14]:


# Check your model
model.summary()


# #### Compile the model for training
# * We will use:
#     - optimizer: Adam optimizer
#     - Loss function: categorical cross-entropy (for multi-class classification)

# In[15]:


opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# #### Initialize hidden state and cell state
# Finally, let's initialize `a0` and `c0` for the LSTM's initial state to be zero. 

# In[16]:


m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))


# #### Train the model
# * Lets now fit the model! 
# * We will turn `Y` into a list, since the cost function expects `Y` to be provided in this format 
#     - `list(Y)` is a list with 30 items, where each of the list items is of shape (60,78). 
#     - Lets train for 100 epochs. This will take a few minutes. 

# In[17]:


model.fit([X, a0, c0], list(Y), epochs=100)


# ## 3 - Generating music
# 
# We now have a trained model which has learned the patterns of the jazz soloist. Lets now use this model to synthesize new music. 
# 
# #### 3.1 - Predicting & Sampling
# 
# <img src="images/music_gen.png" style="width:600;height:400px;">
# 
# At each step of sampling, we will:
# * Take as input the activation '`a`' and cell state '`c`' from the previous state of the LSTM.
# * Forward propagate by one step.
# * Get a new output activation as well as cell state. 
# * The new activation '`a`' can then be used to generate the output using the fully connected layer, `densor`. 
# 
# ##### Initialization
# * We will initialize the following to be zeros:
#     * `x0` 
#     * hidden state `a0` 
#     * cell state `c0` 

# Let's implement the function below to sample a sequence of musical values. 

# In[21]:


def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model


# Run the cell below to define the inference model. 

# In[22]:


inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)


# In[23]:


# Check the inference model
inference_model.summary()


# #### Initialize inference model 

# In[24]:


x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


# Let's implement `predict_and_sample()`. 
# 
# #### Step 1
# * Use your inference model to predict an output given your set of inputs. The output `pred` should be a list of length $T_y$ where each element is a numpy-array of shape (1, n_values).
#  
# #### Step 2
# * Convert `pred` into a numpy array of $T_y$ indices. 
# 
# #### Step 3  
# * Convert the indices into their one-hot vector representations. 

# In[25]:


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices, num_classes=78)
    
    return results, indices


# In[26]:


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))


# #### 3.3 - Generate music 
# 
# Finally, we are ready to generate music. Our RNN generates a sequence of values. The following code generates music by first calling the `predict_and_sample()` function. These values are then post-processed into musical chords (meaning that multiple values or notes can be played at the same time). 
# 
# Most computational music algorithms use some post-processing because it is difficult to generate music that sounds good without such post-processing. The post-processing does things such as clean up the generated audio by making sure the same sound is not repeated too many times, that two successive notes are not too far from each other in pitch, and so on. One could argue that a lot of these post-processing steps are hacks; also, a lot of the music generation literature has also focused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the RNN. But this post-processing does make a huge difference, so let's use it in our implementation as well. 
# 
# Let's make some music! 

# Run the following cell to generate music and record it into our `out_stream`.

# In[27]:


out_stream = generate_music(inference_model)


# As a reference, here is a 30 second audio clip we generated using this algorithm. 

# In[28]:


IPython.display.Audio('./data/30s_trained_model.mp3')


# Congratulations on completing this model and generating a jazz solo! 
