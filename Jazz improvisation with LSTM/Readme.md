# Improvise a Jazz Solo with an LSTM Network

**In this model, we will implement an LSTM (Long Short Term Memory) to generate novel jazz solos in a style representative of a body of performed work.**

We will train our algorithm on a corpus of Jazz music.


#### Details about music
You can informally think of each "value" as a note, which comprises a pitch and duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time. For example, when playing a music piece, you might press down two piano keys at the same time (playing multiple notes at the same time generates what's called a "chord"). But we don't need to worry about the details of music theory for this assignment.

#### Music as a sequence of values
* For the purpose of this assignment, all you need to know is that we will obtain a dataset of values, and will learn an RNN model to generate sequences of values.
* Our music generation system will use 78 unique values.


##### Suggestions are always welcome!
##### Thank you and keep supporting!
