# Horse vs Human Classification
## Best case Accuracy: 94.33% Train Accuracy and 89.06% Validation Accuracy

In this model:
- We are going to differentiate between horse and human
- The contents of the .zip are extracted to the base directory `/tmp/horse-or-human`, which in turn each contain `horses` and `humans` subdirectories.
- We do not explicitly label the images as horses or humans instead we will use `ImageDataGenerator` to read images from subdirectories, and automatically label them from the name of that subdirectory. So, for example, you will have a 'training' directory containing a 'horses' directory and a 'humans' one. ImageGenerator will label the images appropriately for you, reducing a coding step.

In short: The training set is the data that is used to tell the neural network model that 'this is what a horse looks like', 'this is what a human looks like' etc.


#### Summary:
- Downloads dataset on the fly in a `zipfile` and then extracts it for further use
- This is a binary classification problem so it uses `binary_crossentropy` for calculating loss
- Used `ImageDataGenerator` for reading the images from subdirectories, automatically labelling them and scaling the images
- Used `RMSprop` for model optimization.
- Visualize intermediate representations.


##### Suggestions are always welcome!
##### Thank you and keep supporting!
