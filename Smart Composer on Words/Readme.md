# Building Smart Compose with a char n-gram Language Model ([source](https://towardsdatascience.com/gmail-style-smart-compose-using-char-n-gram-language-models-a73c09550447))

In this project, we're going to build a simple and powerful language model from scracth that offers model based prefix search, i.e the text typed in by the user is used as the prefix to predict the next word the user might want to type (in Whatsapp’s case) or user’s search intent (in the Google search case).

### Overview:
I have built a python script that extract n-grams and return them in a python list from a PDF. We will train our model on these n-grams extracted from a PDF. Then we will use the extracted n-grams to train our model which at the end will be able to predict the whole words when given just a few beginning characters of the word. For the python script, stay tuned, I will update it soon.

### Use case:
In another model, we have applied the same model to predict the sentences instead of the complete words.One thing that is common in both the models is that these doesn't require complete words to predict, instead these can make the prediction on sub-words also. Well that's the advantage of an n-gram Language Model.

### Working example:
To predict for a sentence beginning with the words
"The process of", we can even make prediction just by typing "The pro".
Have fun while building your own smart composer.

**Suggestions are always welcome, thank you!**
