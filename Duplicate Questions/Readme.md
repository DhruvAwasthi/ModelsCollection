# Question duplicates
- In this, you will use Siamese networks to natural language processing to build a classifier that will allow you to identify whether two questions asked are same or not.  
- You will process the data first and then pad in a similar way you have done in the previous assignment. Your model will take in the two question embeddings, run them through an LSTM, and then compare the outputs of the two sub networks using cosine similarity.  

- You will be using the Quora question answer dataset to build a model that could identify similar questions. This is a useful task because you don't want to have several versions of the same question posted.
- We select only the question pairs that are duplicate to train the model. We build two batches as input for the Siamese network, and we assume that question one in first batch is a duplicate of question one in second batch and so on, but all other questions in the second batch are not duplicates of question one of first batch.  


- `A Siamese network` is a neural network which uses the same weights while working in tandem on two different input vectors to compute comparable output vectors.
- Siamese networks are important and useful. Many times there are several questions that are already asked in quora, or other platforms and you can use Siamese networks to avoid question duplicates.
- You will use the `TripletLoss` to improve your Siamese model. TripletLoss is composed of two terms. One term utilizes the mean of all the non duplicates, the second utilizes the `closest negative`.  


**Suggestions are always welcome, thank you!**
