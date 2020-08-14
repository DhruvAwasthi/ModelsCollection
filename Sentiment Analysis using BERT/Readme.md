# Sentiment Analysis with Deep Learning using BERT
In this project we're going to use the open-sourced repository- [**transformers**](https://github.com/huggingface/transformers) by HuggingFace to do the sentiment analysis in PyTorch. This repository supports the op-to-op implementation of the official tensorflow code in PyTorch and many new models based on transformers.


## Frequenty Asked Questions (FAQs) about BERT ([source](https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/)):

### What is BERT?
[BERT](https://arxiv.org/abs/1810.04805) is a deep learning model that has given state-of-the-art results on a wide variety of natural language processing tasks. It stands for Bidirectional Encoder Representations for Transformers. It has been pre-trained on Wikipedia and BooksCorpus and requires task-specific fine-tuning.


### What is the model architecture of BERT?
BERT is a multi-layer bidirectional Transformer encoder. There are two models introduced in the paper.

  -  BERT base – 12 layers (transformer blocks), 12 attention heads, and 110 million parameters.
  -  BERT Large – 24 layers, 16 attention heads and, 340 million parameters.


### What is the flow of information of a word in BERT?
A word starts with its embedding representation from the embedding layer. Every layer does some multi-headed attention computation on the word representation of the previous layer to create a new intermediate representation. All these intermediate representations are of the same size. In the figure above, E1 is the embedding representation, T1 is the final output and Trm are the intermediate representations of the same token. In a 12-layers BERT model a token will have 12 intermediate representations.

### What are the tasks BERT has been pre-trained on?
Masked Language Modeling and Next Sentence Prediction

### What downstream tasks can BERT be used for?
BERT can be used for a wide variety of tasks. The two pre-training objectives allow it to be used on any single sequence and sequence-pair tasks without substantial task-specific architecture modifications.

### Which Tokenization strategy is used by BERT?
BERT uses [WordPiece tokenization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf). The vocabulary is initialized with all the individual characters in the language, and then the most frequent/likely combinations of the existing words in the vocabulary are iteratively added.


### What are the optimal values of the hyperparameters used in fine-tuning?
The optimal hyperparameter values are task-specific. But, the authors found that the following range of values works well across all tasks –

  -  Dropout – 0.1
  -  Batch Size – 16, 32
  -  Learning Rate (Adam) – 5e-5, 3e-5, 2e-5
  -  Number of epochs – 3, 4

The authors also observed that large datasets (> 100k labeled samples) are less sensitive to hyperparameter choice than smaller datasets.


### How long does it take to pre-train BERT?
BERT-base was trained on 4 cloud TPUs for 4 days and BERT-large was trained on 16 TPUs for 4 days.


### How long does it take to fine-tune BERT?
For all the fine-tuning tasks discussed in the paper it takes at most 1 hour on a single cloud TPU or a few hours on a GPU.


**Suggestions are always welcome, thank you!**
