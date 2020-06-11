#!/usr/bin/env python
# coding: utf-8

# # Neural Machine Translation

# Let's load all the packages we will need for this model.

# In[1]:


from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1 - Translating human readable dates into machine readable dates
# 
# * The model we will build here could be used to translate from one language to another, such as translating from English to Hindi.
# * We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 

# ### 1.1 - Dataset
# 
# We will train the model on a dataset of 10,000 human readable dates and their equivalent, standardized, machine readable dates. Let's run the following cells to load the dataset and print some examples. 

# In[2]:


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)


# In[3]:


dataset[:10]


# We've loaded:
# - `dataset`: a list of tuples of (human readable date, machine readable date).
# - `human_vocab`: a python dictionary mapping all characters used in the human readable dates to an integer-valued index.
# - `machine_vocab`: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. 
# - `inv_machine_vocab`: the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 
# 
# Let's preprocess the data and map the raw text data into the index values. 
# - We will set Tx=30 
#     - We assume Tx is the maximum length of the human readable date.
#     - If we get a longer input, we would have to truncate it.
# - We will set Ty=10
#     - "YYYY-MM-DD" is 10 characters long.

# In[4]:


Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)


# We now have:
# - `X`: a processed version of the human readable dates in the training set.
#     - Each character in X is replaced by an index (integer) mapped to the character using `human_vocab`. 
#     - Each date is padded to ensure a length of $T_x$ using a special character (< pad >). 
#     - `X.shape = (m, Tx)` where m is the number of training examples in a batch.
# - `Y`: a processed version of the machine readable dates in the training set.
#     - Each character is replaced by the index (integer) it is mapped to in `machine_vocab`. 
#     - `Y.shape = (m, Ty)`. 
# - `Xoh`: one-hot version of `X`
#     - Each index in `X` is converted to the one-hot representation (if the index is 2, the one-hot version has the index position 2 set to 1, and the remaining positions are 0.
#     - `Xoh.shape = (m, Tx, len(human_vocab))`
# - `Yoh`: one-hot version of `Y`
#     - Each index in `Y` is converted to the one-hot representation. 
#     - `Yoh.shape = (m, Tx, len(machine_vocab))`. 
#     - `len(machine_vocab) = 11` since there are 10 numeric digits (0 to 9) and the `-` symbol.

# * Let's also look at some examples of preprocessed training examples. 

# In[5]:


index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])


# ## 2 - Neural machine translation with attention
# 
# * If we had to translate a book's paragraph from French to English, we would not read the whole paragraph, then close the book and translate. 
# * Even during the translation process, we would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English we are writing down. 
# * The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 
# 
# 
# ### 2.1 - Attention mechanism
# 
# In this part, we will implement the attention mechanism presented in the lecture videos. 
# * Here is a figure to remind, how the model works. 
#     * The diagram on the left shows the attention model. 
#     * The diagram on the right shows what one "attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$.
#     * The attention variables $\alpha^{\langle t, t' \rangle}$ are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$). 
# 
# <table>
# <td> 
# <img src="images/attn_model.png" style="width:500;height:500px;"> <br>
# </td> 
# <td> 
# <img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
# </td> 
# </table>
# <caption><center> **Figure 1**: Neural machine translation with attention</center></caption>
# 

# Here are some properties of the model that we may notice: 
# 
# #### Pre-attention and Post-attention LSTMs on both sides of the attention mechanism
# - There are two separate LSTMs in this model (see diagram on the left): pre-attention and post-attention LSTMs.
# - *Pre-attention* Bi-LSTM is the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism.
#     - The attention mechanism is shown in the middle of the left-hand diagram.
#     - The pre-attention Bi-LSTM goes through $T_x$ time steps
# - *Post-attention* LSTM: at the top of the diagram comes *after* the attention mechanism. 
#     - The post-attention LSTM goes through $T_y$ time steps. 
# 
# - The post-attention LSTM passes the hidden state $s^{\langle t \rangle}$ and cell state $c^{\langle t \rangle}$ from one time step to the next. 

# #### An LSTM has both a hidden state and cell state 

# #### Each time step does not use predictions from the previous time step
# * The post-attention LSTM at time 't' only takes the hidden state $s^{\langle t\rangle}$ and cell state $c^{\langle t\rangle}$ as input. 

# #### Computing "energies" $e^{\langle t, t' \rangle}$ as a function of $s^{\langle t-1 \rangle}$ and $a^{\langle t' \rangle}$
# - The definition of "e" as a function of $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$.
#     - "e" is called the "energies" variable.
#     - $s^{\langle t-1 \rangle}$ is the hidden state of the post-attention LSTM
#     - $a^{\langle t' \rangle}$ is the hidden state of the pre-attention LSTM.
#     - $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ are fed into a simple neural network, which learns the function to output $e^{\langle t, t' \rangle}$.
#     - $e^{\langle t, t' \rangle}$ is then used when computing the attention $a^{\langle t, t' \rangle}$ that $y^{\langle t \rangle}$ should pay to $a^{\langle t' \rangle}$.

# ### Implementation Details
#    
# Let's implement this neural translator. We will start by implementing two functions: `one_step_attention()` and `model()`.
# 
# #### one_step_attention
# * The inputs to the one_step_attention at time step $t$ are:
#     - $[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$: all hidden states of the pre-attention Bi-LSTM.
#     - $s^{<t-1>}$: the previous hidden state of the post-attention LSTM 
# * one_step_attention computes:
#     - $[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$: the attention weights
#     - $context^{ \langle t \rangle }$: the context vector:
#     
# $$context^{<t>} = \sum_{t' = 1}^{T_x} \alpha^{<t,t'>}a^{<t'>}\tag{1}$$ 

# Let's implement `one_step_attention()`. 

# In[6]:


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


# In[7]:


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    # For grading purposes, please list 'a' first and 's_prev' second, in this order.
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    
    return context


# #### model
# * `model` first runs the input through a Bi-LSTM to get $[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$. 
# * Then, `model` calls `one_step_attention()` $T_y$ times using a `for` loop.  At each iteration of this loop:
#     - It gives the computed context vector $context^{<t>}$ to the post-attention LSTM.
#     - It runs the output of the post-attention LSTM through a dense layer with softmax activation.
#     - The softmax generates a prediction $\hat{y}^{<t>}$. 

# Let's implement `model()` as explained in figure 1 and the text above. Again, we have defined global layers that will share weights to be used in `model()`.

# In[8]:


n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

# Please note, this is the post attention LSTM cell.  
post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-attention LSTM 
output_layer = Dense(len(machine_vocab), activation=softmax)


# Now you can use these layers $T_y$ times in a `for` loop to generate the outputs, and their parameters will not be reinitialized.

# In[9]:


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model


# Run the following cell to create the model.

# In[10]:


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))


# Let's get a summary of the model:

# In[11]:


model.summary()


# In[12]:


opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])


# #### Define inputs and outputs, and fit the model

# In[13]:


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


# Let's now fit the model and run it for one epoch.

# In[14]:


model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)


# While training we can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives an example of what the accuracies could be if the batch had 2 examples: 
# 
# <img src="images/table.png" style="width:700;height:200px;"> <br>
# <caption><center>Thus, `dense_2_acc_8: 0.89` means that you are predicting the 7th character of the output correctly 89% of the time in the current batch of data. </center></caption>
# 
# 
# We have run this model for longer, and saved the weights. Run the next cell to load our weights.

# In[15]:


model.load_weights('models/model.h5')


# We can now see the results on new examples.

# In[16]:


EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output),"\n")


# ## 3 - Visualizing Attention 
# 
# Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (such as the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can visualize what each part of the output is looking at which part of the input.
# 
# Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed $\alpha^{\langle t, t' \rangle}$ we get this: 
# 
# <img src="images/date_attention.png" style="width:600;height:300px;"> <br>
# <caption><center> **Figure 8**: Full Attention Map</center></caption>
# 
# Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We also see that 9 has been translated as 09 and May has been correctly translated into 05, with the output paying attention to the parts of the input it needs to to make the translation. The year mostly requires it to pay attention to the input's "18" in order to generate "2018." 

# ### 3.1 - Getting the attention weights from the network
# 
# Lets now visualize the attention values in your network. We'll propagate an example through the network, then visualize the values of $\alpha^{\langle t, t' \rangle}$. 
# 
# To figure out where the attention values are located, let's start by printing a summary of the model .

# In[17]:


model.summary()


# Navigate through the output of `model.summary()` above. We can see that the layer named `attention_weights` outputs the `alphas` of shape (m, 30, 1) before `dot_2` computes the context vector for every time step $t = 0, \ldots, T_y-1$. Let's get the attention weights from this layer.
# 
# The function `attention_map()` pulls out the attention values from your model and plots them.

# In[18]:


attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);


# On the generated plot we can observe the values of the attention weights for each character of the predicted output. 
# 
# In the date translation application, we will observe that most of the time attention helps predict the year, and doesn't have much impact on predicting the day or month.

# ### Congratulations!
# 
# ## Here's what you should remember
# 
# - Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation. 
# - An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
# - A network using an attention mechanism can translate from inputs of length $T_x$ to outputs of length $T_y$, where $T_x$ and $T_y$ can be different. 
# - We can visualize attention weights $\alpha^{\langle t,t' \rangle}$ to see what the network is paying attention to while generating each output.
