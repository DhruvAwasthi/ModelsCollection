# Happy vs Sad Face Classification
## Best case Accuracy: 100% on Train Set
**Understanding the dataset:**
The dataset contains 80 images of which 40 are happy and 40 are sad faces.

#### Summary:
- Downloads dataset on the fly in a `zipfile` and then extracts it for further use
- This is a binary classification problem so it uses `binary_crossentropy` for calculating loss
- Used `ImageDataGenerator` for scaling the input images
- Used `RMSprop` for model optimization.
- Used `callbacks` for stopping training when accuracy achieved 99.8%
- Used three convolutional layers.


**Suggestions are always welcome, thank you!**
