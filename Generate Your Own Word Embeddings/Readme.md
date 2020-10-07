# Generate Your Own Word Embeddings

In this, you will practice how to compute word embeddings and use them for sentiment analysis.
- To implement sentiment analysis, you can go beyond counting the number of positive words and negative words.
- You can find a way to represent each word numerically, by a vector.
- The vector could then represent syntactic (i.e. parts of speech) and semantic (i.e. meaning) structures.

In this, you will explore a classic way of generating word embeddings or representations.
- You will implement a famous model called the continuous bag of words (CBOW) model.
- In continuous bag of words (CBOW) modeling, we try to predict the center word given a few context words (the words around the center word).  

Let's take a look at the following sentence:
>**'I am happy because I am learning'**.
- For example, if you were to choose a context half-size of say C = 2, then you would try to predict the word **happy** given the context that includes 2 words before and 2 words after the center word:

##### C words before: [I, am]

##### C words after: [because, I]

- In other words:

context = [I,am, because, I]
target = happy

Knowing how to train these models will give you a better understanding of word vectors, which are building blocks to many applications in natural language processing.  

**Suggestions are always welcome, thank you!**
