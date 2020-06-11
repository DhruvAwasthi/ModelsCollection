#!/usr/bin/env python
# coding: utf-8

# # Emojify! 
# 
# * You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️).
# 
# #### What you'll build
# 1. In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings.
# 2. Then you will build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM. 

# Let's get started! Run the following cell to load the package you are going to use. 

# In[1]:


import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1 - Baseline model: Emojifier-V1
# 
# ### 1.1 - Dataset EMOJISET
# 
# Let's start by building a simple baseline classifier. 
# 
# We have a tiny dataset (X, Y) where:
# - X contains 127 sentences (strings).
# - Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence.
# 
# <img src="images/data_set.png" style="width:700px;height:300px;">
# <caption><center> **Figure 1**: EMOJISET - a classification problem with 5 classes. A few examples of sentences are given here. </center></caption>
# 
# Let's load the dataset using the code below. We split the dataset between training (127 examples) and testing (56 examples).

# In[2]:


X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')


# In[10]:


maxLen = len(max(X_train, key=len).split())


# Run the following cell to print sentences from X_train and corresponding labels from Y_train. 
# * Note that due to the font used by iPython notebook, the heart emoji may be colored black rather than red.

# In[11]:


for idx in range(10):
    print(X_train[idx], label_to_emoji(Y_train[idx]))


# ### 1.2 - Overview of the Emojifier-V1
# 
# In this part, we are going to implement a baseline model called "Emojifier-v1".  
# 
# <center>
# <img src="images/image_1.png" style="width:900px;height:300px;">
# <caption><center> **Figure 2**: Baseline model (Emojifier-V1).</center></caption>
# </center>
# 
# 
# #### Inputs and outputs
# * The input of the model is a string corresponding to a sentence (e.g. "I love you). 
# * The output will be a probability vector of shape (1,5), (there are 5 emojis to choose from).
# * The (1,5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability.

# #### One-hot encoding
# * To get our labels into a format suitable for training a softmax classifier, lets convert $Y$ from its current shape  $(m, 1)$ into a "one-hot representation" $(m, 5)$, 
#     * Each row is a one-hot vector giving the label of one example.
#     * Here, `Y_oh` stands for "Y-one-hot" in the variable names `Y_oh_train` and `Y_oh_test`: 

# In[12]:


Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)


# Let's see what `convert_to_one_hot()` did. Feel free to change `index` to print out different values. 

# In[13]:


idx = 50
print(f"Sentence '{X_train[50]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")


# All the data is now ready to be fed into the Emojify-V1 model. Let's implement the model!

# ### 1.3 - Implementing Emojifier-V1
# 
# As shown in Figure 2 (above), the first step is to:
# * Convert each word in the input sentence into their word vector representations.
# * Then take an average of the word vectors. 
# * Similar to the previous exercise, we will use pre-trained 50-dimensional GloVe embeddings. 
# 
# Run the following cell to load the `word_to_vec_map`, which contains all the vector representations.

# In[14]:


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt')


# We've loaded:
# - `word_to_index`: dictionary mapping from words to their indices in the vocabulary 
#     - (400,001 words, with the valid indices ranging from 0 to 400,000)
# - `index_to_word`: dictionary mapping from indices to their corresponding words in the vocabulary
# - `word_to_vec_map`: dictionary mapping words to their GloVe vector representation.
# 
# Run the following cell to check if it works.

# In[16]:


word = "cucumber"
idx = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])


# Let's implement `sentence_to_avg()`. We will need to carry out two steps:
# 1. Convert every sentence to lower-case, then split the sentence into a list of words. 
# 2. For each word in the sentence, access its GloVe representation.
#     * Then take the average of all of these word vectors.

# In[24]:


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros((len(words), 1))
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    total = 0
    for w in words:
        total += word_to_vec_map[w]
    avg = total / len(words)
    
    return avg


# In[25]:


avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = \n", avg)


# #### Model
# Let's implement the `model()` function described in Figure (2). 
# 
# * The equations we need to implement in the forward pass and to compute the cross-entropy cost are below:
# * The variable $Y_{oh}$ ("Y one hot") is the one-hot encoding of the output labels. 
# 
# $$ z^{(i)} = W . avg^{(i)} + b$$
# 
# $$ a^{(i)} = softmax(z^{(i)})$$
# 
# $$ \mathcal{L}^{(i)} = - \sum_{k = 0}^{n_y - 1} Y_{oh,k}^{(i)} * log(a^{(i)}_k)$$

# In[29]:


def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    n_y = 5                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    
    # Optimization loop
    for t in range(num_iterations): # Loop over the number of iterations
        for i in range(m):          # Loop over the training examples
        
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(Y_oh[i] * np.log(a))
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

    return pred, W, b


# In[30]:


print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)

X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
 'Lets go party and drinks','Congrats on the new job','Congratulations',
 'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
 'You totally deserve this prize', 'Let us go play football',
 'Are you down for football this afternoon', 'Work hard play harder',
 'It is suprising how people can be dumb sometimes',
 'I am very disappointed','It is the best day in my life',
 'I think I will end up alone','My life is so boring','Good job',
 'Great so awesome'])

print(X.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(type(X_train))


# Run the next cell to train the model and learn the softmax parameters (W,b). 

# In[31]:


pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)


# ### 1.4 - Examining test set performance 
# 
# * Note that the `predict` function used here is defined in emo_utils.py.

# In[32]:


print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)


# Great! Our model has pretty high accuracy on the training set. 

# * Random guessing would have had 20% accuracy given that there are 5 classes. (1/5 = 20%).
# * This is pretty good performance after training on only 127 examples. 
# 
# 
# #### The model matches emojis to relevant words
# In the training set, the algorithm saw the sentence 
# >"*I love you*" 
# 
# with the label ❤️. 
# * We can check that the word "adore" does not appear in the training set. 
# * Nonetheless, lets see what happens if we write "*I adore you*."
# 

# In[33]:


X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)


# Amazing! 
# * Because *adore* has a similar embedding as *love*, the algorithm has generalized correctly even to a word it has never seen before. 
# * Words such as *heart*, *dear*, *beloved* or *adore* have embedding vectors similar to *love*. 
# 
# #### Word ordering isn't considered in this model
# * Note that the model doesn't get the following sentence correct:
# >"not feeling happy" 
# 
# * This algorithm ignores word ordering, so is not good at understanding phrases like "not happy." 
# 
# #### Confusion matrix
# * Printing the confusion matrix can also help understand which classes are more difficult for our model. 
# * A confusion matrix shows how often an example whose label is one class ("actual" class) is mislabeled by the algorithm with a different class ("predicted" class).

# In[34]:


print(Y_test.shape)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)


# ## 2 - Emojifier-V2: Using LSTMs in Keras: 
# 
# Let's build an LSTM model that takes word **sequences** as input!
# * This model will be able to account for the word ordering. 
# * Emojifier-V2 will continue to use pre-trained word embeddings to represent words.
# * We will feed word embeddings into an LSTM.
# * The LSTM will learn to predict the most appropriate emoji. 
# 
# Run the following cell to load the Keras packages.

# In[35]:


import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


# ### 2.1 - Overview of the model
# 
# Here is the Emojifier-v2 we will implement:
# 
# <img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
# <caption><center> **Figure 3**: Emojifier-V2. A 2-layer LSTM sequence classifier. </center></caption>
# 
# 

# ### 2.2 Keras, padding and mini-batching 

# ### 2.3 - The Embedding layer
# 
# * In Keras, the embedding matrix is represented as a "layer".
# * The embedding matrix maps word indices to embedding vectors.
#     
# #### Using and updating pre-trained embeddings
# * In this part, we will initialize the Embedding layer with the GloVe 50-dimensional vectors. 

# #### Inputs and outputs to the embedding layer
# 
# * The `Embedding()` layer's input is an integer matrix of size **(batch size, max input length)**. 
#     * This input corresponds to sentences converted into lists of indices (integers).
#     * The largest integer (the highest word index) in the input should be no larger than the vocabulary size.
# * The embedding layer outputs an array of shape (batch size, max input length, dimension of word vectors).
# 
# * The figure shows the propagation of two example sentences through the embedding layer. 
#     * Both examples have been zero-padded to a length of `max_len=5`.
#     * The word embeddings are 50 units in length.
#     * The final dimension of the representation is  `(2,max_len,50)`. 
# 
# <img src="images/embedding1.png" style="width:700px;height:250px;">
# <caption><center> **Figure 4**: Embedding layer</center></caption>

# #### Prepare the input sentences
# Let's implement `sentences_to_indices`, which processes an array of sentences (X) and returns inputs to the embedding layer.

# In[36]:


for idx, val in enumerate(["I", "like", "learning"]):
    print(idx,val)


# In[41]:


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
    
    return X_indices


# Run the following cell to check what `sentences_to_indices()` does, and check the results.

# In[42]:


X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =\n", X1_indices)


# #### Build embedding layer
# 
# * Let's build the `Embedding()` layer in Keras, using pre-trained word vectors. 
# * The embedding layer takes as input a list of word indices.
#     * `sentences_to_indices()` creates these word indices.
# * The embedding layer will return the word embeddings for a sentence. 
# 
# Let's implement `pretrained_embedding_layer()` with these steps:
# 1. Initialize the embedding matrix as a numpy array of zeros.
# 2. Fill in each row of the embedding matrix with the vector representation of a word
# 3. Define the Keras embedding layer. 
#         * If you were to set `trainable = True`, then it will allow the optimization algorithm to modify the         values of the word embeddings.
#         * In this case, we don't want the model to modify the word embeddings.
# 4. Set the embedding weights to be equal to the embedding matrix.

# In[43]:


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[44]:


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# ## 2.3 Building the Emojifier-V2
# 
# Lets now build the Emojifier-V2 model. 
# * We feed the embedding layer's output to an LSTM network. 
# 
# <img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
# <caption><center> **Figure 3**: Emojifier-v2. A 2-layer LSTM sequence classifier. </center></caption>
# 
# 
# Let's implement `Emojify_V2()`, which builds a Keras graph of the architecture shown in Figure 3. 
# * The model takes as input an array of sentences of shape (`m`, `max_len`, ) defined by `input_shape`. 
# * The model outputs a softmax probability vector of shape (`m`, `C = 5`). 

# In[53]:


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)   
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation("softmax")(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)
    
    return model


# Run the following cell to create the model and check its summary. Because all sentences in the dataset are less than 10 words, we chose `max_len = 10`.  We should see the architecture, it uses "20,223,927" parameters, of which 20,000,050 (the word embeddings) are non-trainable, and the remaining 223,877 are. Because our vocabulary size has 400,001 words (with valid indices from 0 to 400,000) there are 400,001\*50 = 20,000,050 non-trainable parameters. 

# In[54]:


model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()


# In[55]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# It's time to train your model. The Emojifier-V2 `model` takes as input an array of shape (`m`, `max_len`) and outputs probability vectors of shape (`m`, `number of classes`). We thus have to convert X_train (array of sentences as strings) to X_train_indices (array of sentences as list of word indices), and Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors).

# In[56]:


X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)


# In[57]:


model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)


# In[58]:


X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)


# Run the cell below to see the mislabelled examples. 

# In[59]:


# This code allows to see the mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())


# Now we can try it on our own example:

# In[60]:


# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


# ### Congratulations!
# 
# You have completed this model! ❤️❤️❤️
# 
# 
# ## What we should remember
# - If we have an NLP task where the training set is small, using word embeddings can help the algorithm significantly. 
# - Word embeddings allow our model to work on words in the test set that may not even appear in the training set. 
# - Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details:
#     - To use mini-batches, the sequences need to be **padded** so that all the examples in a mini-batch have the **same length**. 
#     - An `Embedding()` layer can be initialized with pretrained values. 
#         - These values can be either fixed or trained further on your dataset. 
#         - If however your labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.   
#     - `LSTM()` has a flag called `return_sequences` to decide if we would like to return every hidden states or only the last one. 
#     - We can use `Dropout()` right after `LSTM()` to regularize your network. 

# 
# #### Input sentences:
# ```Python
# "Congratulations on finishing this assignment and building an Emojifier."
# "We hope you're happy with what you've accomplished in this notebook!"
# ```
# #### Output emojis:
# # 😀😀😀😀😀😀
