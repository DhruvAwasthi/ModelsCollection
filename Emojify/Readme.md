# Emojify - Suggesting Emojis for Text
## Best case Accuracy: 98.0% Accuracy on Train Set and >90% Accuracy on Test Set
We are going to use word vector representations to build an Emojifier.

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that.
So rather than writing:
>"Congratulations on the promotion! Let's get coffee and talk. Love you!"   

The emojifier can automatically turn this into:
>"Congratulations on the promotion! 👍 Let's get coffee and talk. ☕️ Love you! ❤️"

#### Using word vectors to improve emoji lookups
* In many emoji interfaces, you need to remember that ❤️ is the "heart" symbol rather than the "love" symbol.
    * In other words, you'll have to remember to type "heart" to find the desired emoji, and typing "love" won't bring up that symbol.
* We can make a more flexible emoji interface by using word vectors!
* When using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate additional words in the test set to the same emoji.
    * This works even if those additional words don't even appear in the training set.
    * This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.

### The Embedding layer

  * In Keras, the embedding matrix is represented as a "layer".    * The embedding matrix maps word indices to embedding vectors.
      * The word indices are positive integers.
      * The embedding vectors are dense vectors of fixed size.
      * When we say a vector is "dense", in this context, it means that most of the values are non-zero.  As a counter-example, a one-hot encoded vector is not "dense."
  * The embedding matrix can be derived in two ways:
      * Training a model to derive the embeddings from scratch.
      * Using a pretrained embedding


**Suggestions are always welcome, thank you!**
