# Transformer Summarizer
- In this, you will explore summarization using the transformer model. You will implement the transformer decoder from scratch. And then use this to generate the summary of an article given to it.
- It is currently the state-of-the-art practice to use transformers.

### A Quck Overview:
- Summarization is an important task in natural language processing and could be useful for a consumer enterprise. For example, bots can be used to scrape articles, summarize them, and then you can use sentiment analysis to identify the sentiment about certain stocks. Anyways who wants to read an article or a long email today, when you can build a transformer to summarize text for you haha.
- As you know, language models only predict the next word, they have no notion of inputs. To create a single input suitable for a language model, we concatenate inputs with targets putting a separator in between. We also need to create a mask -- with 0s at inputs and 1s at targets -- so that the model is not penalized for mis-predicting the article and only focuses on the summary.


**Happy learning!**
