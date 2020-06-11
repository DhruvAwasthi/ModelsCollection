# Character Level Language Model - Dinosaurus Island

- In this model, our algorithm will learn the different name patterns, and randomly generate new names.
- We have collected a list of all the dinosaur names we could find, and compiled them into this [dataset](dinos.txt). To create new dinosaur names, we will build a character level language model to generate new names.

**Understanding the dataset:**
- There are 19909 total characters and 27 unique characters in data.
- The characters are a-z (26 characters) plus the "\n" (or newline character).
- The newline character "\n" plays a role similar to the <EOS> (or "End of sentence") token
- Here, "\n" indicates the end of the dinosaur name rather than the end of a sentence.

---

# Writing like Shakespeare
A similar (but more complicated) task is to generate Shakespeare poems. Instead of learning from a dataset of Dinosaur names we can use a collection of Shakespearian poems. Using LSTM cells, we can learn longer term dependencies that span many characters in the text--e.g., where a character appearing somewhere a sequence can influence what should be a different character much much later in the sequence. These long term dependencies were less important with dinosaur names, since the names were quite short.

##### Suggestions are always welcome!
##### Thank you and keep supporting!
