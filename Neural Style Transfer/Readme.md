# Deep Learning & Art: Neural Style Transfer

**In this model we implement the neura style transfer algorithm and generate novel artistic images using the algorithm.**

- Most of the algorithms We've studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, we'll optimize a cost function to get pixel values!
- Neural Style Transfer (NST) is one of the most fun techniques in deep learning. It merges two images, namely: a **"content" image (C) and a "style" image (S), to create a "generated" image (G**).
- The generated image G combines the "content" of the image C with the "style" of image S.

---

### Transfer Learning
- Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.
- Following the [original NST paper](https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

---

### Neural Style Transfer (NST)
We will build the Neural Style Transfer (NST) algorithm in three steps:
- Build the content cost function
- Build the style cost function
- Put it together.


**Suggestions are always welcome, thank you!**
