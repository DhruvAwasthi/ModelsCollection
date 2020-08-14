# Neural Machine Translation

In this model:
- We will build a Neural Machine Translation (NMT) model to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25").
- We will do this using an attention model, one of the most sophisticated sequence-to-sequence models.

## Translating human readable dates into machine readable dates

* The model we will build here could be used to translate from one language to another, such as translating from English to Hindi.
* However, language translation requires massive datasets and usually takes days of training on GPUs.
* As an experiment with these models without using massive datasets, we will perform a simpler "date translation" task.
* The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*)
* The network will translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*).
* We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.


**Suggestions are always welcome, thank you!**
