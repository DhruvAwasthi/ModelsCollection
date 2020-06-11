#!/usr/bin/env python
# coding: utf-8

# # Deep Learning & Art: Neural Style Transfer

# In[1]:


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1 - Understandin the Problem Statement
# 
# Neural Style Transfer (NST) is one of the most fun techniques in deep learning. It merges two images, namely: a **"content" image (C) and a "style" image (S), to create a "generated" image (G**). 
# 
# The generated image G combines the "content" of the image C with the "style" of image S. 
# 
# Let's see how you can do this. 

# ## 2 - Transfer Learning
# 
# Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning. 
# 
# Following the [original NST paper](https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers). 
# 
# Run the following code to load parameters from the VGG model.

# In[2]:


pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)


# * The model is stored in a python dictionary.  
# * The python dictionary contains key-value pairs for each layer.  
# * The 'key' is the variable name and the 'value' is a tensor for that layer. 

# ## 3 - Neural Style Transfer (NST)
# 
# We will build the Neural Style Transfer (NST) algorithm in three steps:
# 
# - Build the content cost function $J_{content}(C,G)$
# - Build the style cost function $J_{style}(S,G)$
# - Put it together to get $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$. 
# 
# ### 3.1 - Computing the content cost
# 
# In our running example, the content image C will be the picture of the Louvre Museum in Paris. Run the code below to see a picture of the Louvre.

# In[3]:


content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);


# The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.
# 
# ** 3.1.1 - Make generated image G match the content of image C**
# 
# #### Shallower versus deeper layers
# * The shallower layers of a ConvNet tend to detect lower-level features such as edges and simple textures.
# * The deeper layers tend to detect higher-level features such as more complex textures as well as object classes. 
# 
# #### Choose a "middle" activation layer $a^{[l]}$
# We would like the "generated" image G to have similar content as the input image C. Suppose we have chosen some layer's activations to represent the content of an image. 
# * In practice, we'll get the most visually pleasing results if we choose a layer in the **middle** of the network--neither too shallow nor too deep. 
# 
# #### Forward propagate image "C"
# * Set the image C as the input to the pretrained VGG network, and run forward propagation.  
# * Let $a^{(C)}$ be the hidden layer activations in the layer you had chosen. (In lecture, we had written this as $a^{[l](C)}$, but here we'll drop the superscript $[l]$ to simplify the notation.) This will be an $n_H \times n_W \times n_C$ tensor.
# 
# #### Forward propagate image "G"
# * Repeat this process with the image G: Set G as the input, and run forward progation. 
# * Let $a^{(G)}$ be the corresponding hidden layer activation. 
# 
# #### Content Cost Function $J_{content}(C,G)$
# We will define the content cost function as:
# 
# $$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $$
# 
# * Here, $n_H, n_W$ and $n_C$ are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost. 
# * For clarity, note that $a^{(C)}$ and $a^{(G)}$ are the 3D volumes corresponding to a hidden layer's activations. 
# * In order to compute the cost $J_{content}(C,G)$, it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.
# * Technically this unrolling step isn't needed to compute $J_{content}$, but it will be good practice for when you do need to carry out a similar operation later for computing the style cost $J_{style}$.

# Let's compute the "content cost" using TensorFlow. 
# 
# The 3 steps to implement this function are:
# 1. Retrieve dimensions from `a_G` 
# 2. Unroll `a_C` and `a_G`
# 3. Compute the content cost:

# In[10]:


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = (1 / ( 4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content


# In[11]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))


# ### 3.2 - Computing the style cost
# 
# For our running example, we will use the following style image: 

# In[12]:


style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image);


# This was painted in the style of *[impressionism](https://en.wikipedia.org/wiki/Impressionism)*.
# 
# Lets see how we can now define a "style" cost function $J_{style}(S,G)$. 

# ### 3.2.1 - Style matrix
# 
# #### Gram matrix
# * The style matrix is also called a "Gram matrix." 
# * In linear algebra, the Gram matrix G of a set of vectors $(v_{1},\dots ,v_{n})$ is the matrix of dot products, whose entries are ${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j})  }$. 
# * In other words, $G_{ij}$ compares how similar $v_i$ is to $v_j$: If they are highly similar, you would expect them to have a large dot product, and thus for $G_{ij}$ to be large. 
# 
# #### Two meanings of the variable $G$
# * Note that there is an unfortunate collision in the variable names used here. We are following common terminology used in the literature. 
# * $G$ is used to denote the Style matrix (or Gram matrix) 
# * $G$ also denotes the generated image. 
# * For this model, we will use $G_{gram}$ to refer to the Gram matrix, and $G$ to denote the generated image.

# 
# #### Compute $G_{gram}$
# In Neural Style Transfer (NST), we can compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose:
# 
# $$\mathbf{G}_{gram} = \mathbf{A}_{unrolled} \mathbf{A}_{unrolled}^T$$
# 
# #### $G_{(gram)i,j}$: correlation
# The result is a matrix of dimension $(n_C,n_C)$ where $n_C$ is the number of filters (channels). The value $G_{(gram)i,j}$ measures how similar the activations of filter $i$ are to the activations of filter $j$. 
# 
# #### $G_{(gram),i,i}$: prevalence of patterns or textures
# * The diagonal elements $G_{(gram)ii}$ measure how "active" a filter $i$ is. 
# * For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{(gram)ii}$ measures how common  vertical textures are in the image as a whole.
# * If $G_{(gram)ii}$ is large, this means that the image has a lot of vertical texture. 
# 
# 
# By capturing the prevalence of different types of features ($G_{(gram)ii}$), as well as how much different features occur together ($G_{(gram)ij}$), the Style matrix $G_{gram}$ measures the style of an image. 

# Let's implement a function that computes the Gram matrix of a matrix A. 
# * The formula is: The gram matrix of A is $G_A = AA^T$. 

# In[13]:


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


# In[14]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = \n" + str(GA.eval()))


# ### 3.2.2 - Style cost

# Our goal will be to minimize the distance between the Gram matrix of the "style" image S and the gram matrix of the "generated" image G. 
# * For now, we are using only a single hidden layer $a^{[l]}$.  
# * The corresponding style cost for this layer is defined as: 
# 
# $$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2\tag{2} $$
# 
# * $G_{gram}^{(S)}$ Gram matrix of the "style" image.
# * $G_{gram}^{(G)}$ Gram matrix of the "generated" image.
# * Remember, this cost is computed using the hidden layer activations for a particular hidden layer in the network $a^{[l]}$
# 

# Let's compute the style cost for a single layer. 
# 
# The 3 steps to implement this function are:
# 1. Retrieve dimensions from the hidden layer activations a_G 
# 2. Unroll the hidden layer activations a_S and a_G into 2D matrices
# 3. Compute the Style matrix of the images S and G. 
# 4. Compute the Style cost.

# In[27]:


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1 / (4 * n_C * n_C * n_H * n_H * n_W * n_W)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer


# In[28]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))


# ### 3.2.3 Style Weights
# 
# * So far we have captured the style from only one layer. 
# * We'll get better results if we "merge" style costs from several different layers. 
# * Each layer will be given weights ($\lambda^{[l]}$) that reflect how much each layer will contribute to the style.
# * By default, we'll give each layer equal weight, and the weights add up to 1.  ($\sum_{l}^L\lambda^{[l]} = 1$)

# In[29]:


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# We can combine the style costs for different layers as follows:
# 
# $$J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)$$
# 
# where the values for $\lambda^{[l]}$ are given in `STYLE_LAYERS`. 
# 

# Let's implement `compute style cost`
# 
# * We've implemented a compute_style_cost(...) function. 
# * It calls our `compute_layer_style_cost(...)` several times, and weights their results using the values in `STYLE_LAYERS`. 
# 
# #### Description of `compute_style_cost`
# For each layer:
# * Select the activation (the output tensor) of the current layer.
# * Get the style of the style image "S" from the current layer.
# * Get the style of the generated image "G" from the current layer.
# * Compute the "style cost" for the current layer
# * Add the weighted style cost to the overall style cost (J_style)
# 
# Once we're done with the loop:  
# * Return the overall style cost.

# In[30]:


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


# ### 3.3 - Defining the total cost to optimize

# Finally, let's create a cost function that minimizes both the style and the content cost. The formula is: 
# 
# $$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$
# 
# Let's implement the total cost function which includes both the content cost and the style cost. 

# In[31]:


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha * J_content + beta * J_style
    
    return J


# In[32]:


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


# ## 4 - Solving the optimization problem

# Finally, let's put everything together to implement Neural Style Transfer!
# 
# 
# Here's what the program will have to do:
# 
# 1. Create an Interactive Session
# 2. Load the content image 
# 3. Load the style image
# 4. Randomly initialize the image to be generated 
# 5. Load the VGG19 model
# 7. Build the TensorFlow graph:
#     - Run the content image through the VGG19 model and compute the content cost
#     - Run the style image through the VGG19 model and compute the style cost
#     - Compute the total cost
#     - Define the optimizer and the learning rate
# 8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
# 
# Lets go through the individual steps in detail. 

# #### Interactive Sessions
# 
# We've previously implemented the overall cost $J(G)$. We'll now set up TensorFlow to optimize this with respect to $G$. 
# * Unlike a regular session, the "Interactive Session" installs itself as the default session to build a graph.  
# * This allows us to run variables without constantly needing to refer to the session object (calling "sess.run()"), which simplifies the code.  
# 
# #### Start the interactive session.

# In[33]:


# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()


# #### Content image
# Let's load, reshape, and normalize our "content" image (the Louvre museum picture):

# In[34]:


content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)


# #### Style image
# Let's load, reshape and normalize our "style" image (Claude Monet's painting):

# In[35]:


style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)


# #### Generated image correlated with content image
# Now, we initialize the "generated" image as a noisy image created from the content_image.
# 
# * The generated image is slightly correlated with the content image.
# * By initializing the pixels of the generated image to be mostly noise but slightly correlated with the content image, this will help the content of the "generated" image more rapidly match the content of the "content" image. 
# * Feel free to look in `nst_utils.py` to see the details of `generate_noise_image(...)`.

# In[36]:


generated_image = generate_noise_image(content_image)
imshow(generated_image[0]);


# #### Load pre-trained VGG19 model
# Next, as explained in part (2), let's load the VGG19 model.

# In[37]:


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# #### Content Cost
# 
# To get the program to compute the content cost, we will now assign `a_C` and `a_G` to be the appropriate hidden layer activations. We will use layer `conv4_2` to compute the content cost. The code below does the following:
# 
# 1. Assign the content image to be the input to the VGG model.
# 2. Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
# 3. Set a_G to be the tensor giving the hidden layer activation for the same layer. 
# 4. Compute the content cost using a_C and a_G.
# 
# **Note**: At this point, a_G is a tensor and hasn't been evaluated. It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.

# In[38]:


# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)


# #### Style cost

# In[39]:


# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


# ### Total cost
# * Now that you have J_content and J_style, compute the total cost J by calling `total_cost()`. 
# * Use `alpha = 10` and `beta = 40`.

# In[1]:


J = total_cost(J_content, J_style, alpha = 10, beta = 40)


# ### Optimizer
# 
# * Use the Adam optimizer to minimize the total cost `J`.
# * Use a learning rate of 2.0.  

# In[42]:


# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)


# ### implement the model
# 
# * Implement the model_nn() function.  
# * The function **initializes** the variables of the tensorflow graph, 
# * **assigns** the input image (initial generated image) as the input of the VGG19 model 
# * and **runs** the `train_step` tensor (it was created in the code above this function) for a large number of steps.

# In[44]:


def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


# Run the following cell to generate an artistic image. 

# In[45]:


model_nn(sess, generated_image)


# We're done! After running this, we should see something the image presented below on the right:
# 
# <img src="images/louvre_generated.png" style="width:800px;height:300px;">
# 
# We didn't want you to wait too long to see an initial result, and so had set the hyperparameters accordingly. To get the best looking results, running the optimization algorithm longer (and perhaps with a smaller learning rate) might work better. After completing and submitting this assignment, we encourage you to come back and play more with this notebook, and see if you can generate even better looking images. 

# Here are few other examples:
# 
# - The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
# <img src="images/perspolis_vangogh.png" style="width:750px;height:300px;">
# 
# - The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
# <img src="images/pasargad_kashi.png" style="width:750px;height:300px;">
# 
# - A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
# <img src="images/circle_abstract.png" style="width:750px;height:300px;">

# ## What we should remember
# - Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
# - It uses representations (hidden layer activations) based on a pretrained ConvNet. 
# - The content cost function is computed using one hidden layer's activations.
# - The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
# - Optimizing the total cost function results in synthesizing new images. 
