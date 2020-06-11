#Residual Netwroks (ResNets) on SIGNS dataset
## Best case Acuracy: 86.67% test accuracy

In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks (ResNets) allow you to train much deeper networks than were previously practically feasible.

**Understanding the dataset:**
SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5. It contains a 1200 images of which we use 1080 images as train examples and 120 images as validation examples.

#### Summary:

`The problem of very deep neural networks`:

In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.

- The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output).
- However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow.
- More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values).
- During training, you might therefore see the magnitude (or norm) of the gradient for the shallower layers decrease to zero very rapidly as training proceeds:

We are now going to solve this problem by building a Residual Network!

##### Suggestions are always welcome!
##### Thank you and keep supporting!
