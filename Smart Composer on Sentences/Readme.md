# Building Smart Compose with a char n-gram Language Model ([source](https://towardsdatascience.com/gmail-style-smart-compose-using-char-n-gram-language-models-a73c09550447))

In this project, we're going to build a simple and powerful language model from scracth that offers model based prefix search, i.e the text typed in by the user is used as the prefix to predict the next word the user might want to type (in Whatsapp’s case) or user’s search intent (in the Google search case).

### Overview:
We will train our model on the PDF titled E_376.pdf. Our code will parse all the contents of the PDF, do the preprocessing to remove unnecessary information and save the content to a .txt filt, all on the fly. Then we will use the parsed content to train our model which at the end will be able to predict the whole sentence when given just a beginning word of the sentence

### Use case:
In another model, we have applied the same model to predict the words instead of the complete sentences.One thing that is common in both the models is that these doesn't require complete words to predict, instead these can make the prediction on sub-words also. Well that's the advantage of an n-gram Language Model.

### Working example:
To predict for a sentence beginning with the words
"The process of", we can even make prediction just by typing "The pro".
Have fun while building your own smart composer.

**Suggestions are always welcome, thank you!**
