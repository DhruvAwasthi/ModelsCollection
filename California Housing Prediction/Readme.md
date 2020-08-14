# California Housing Prediciton Model
## Bese case Standard Deviation: 1634.46, using Support Vector Machine Regressor

#### Summary:
- Downloading and loading raw data on the fly using `urllib` and `pandas`.
- Used various techniques to get the most out of data such as `StratifiedShuffleSplit`, `Feature Engineering` , `Pipeline`, `LabelEncoder`, `GridSearchCV` and `Correlation Matrix`
- Tried various models to choose the best that fits `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `SVR (Support Vector Machine Regressor)`
- Used various metrics to evaluate the model such as `CrossValidation`, `MSE`, `RMSE`
- Tried possibility to save and load pretrained models for later use.


**STRATIFIED RANDOM SAMPLING:**

When completing analysis or research on a group of entities with similar characteristics, a researcher may find that the population size is too large for which to complete research. To save time and money, an analyst may take on a more feasible approach by selecting a small group from the population. The small group is referred to as a sample size, which is a subset of the population that is used to represent the entire population. A sample may be selected from a population through a number of ways, one of which is the stratified random sampling method. A stratified random sampling involves dividing the entire population into homogeneous groups called strata (plural for stratum). Random samples are then selected from each stratum.

SOURCE: https://www.investopedia.com/terms/stratified_random_sampling.asp

**STANDARD CORRELATION COEFFICIENT:**

The correlation coefficient is a statistical measure of the strength of the relationship between the relative movements of two variables. The values range between -1.0 and 1.0. A calculated number greater than 1.0 or less than -1.0 means that there was an error in the correlation measurement. A value of exactly 1.0 means there is a perfect positive relationship between the two variables. For a positive increase in one variable, there is also a positive increase in the second variable. A value of -1.0 means there is a perfect negative relationship between the two variables. This shows that the variables move in opposite directions - for a positive increase in one variable, there is a decrease in the second variable. If the correlation between two variables is 0, there is no linear relationship between them.

**k-FOLD CROSS VALIDATION:**

Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model. The general procedure is as follows:

   1. Shuffle the dataset randomly.
   2. Split the dataset into k groups
   3. For each unique group:
        a) Take the group as a hold out or test data set
        b) Take the remaining groups as a training data set
        c) Fit a model on the training set and evaluate it on the test set
        d) Retain the evaluation score and discard the model
   4. Summarize the skill of the model using the sample of model evaluation scores

SOURCE: https://machinelearningmastery.com/k-fold-cross-validation/

Cross validation allows you to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation).

**RANDOMIZED SEARCH:**

The grid search approach is fine when you are exploring the relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead. This class can be sued in much the same way as the GridSearchCV class, but instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

**Tip:** Another way to fine-tune your model is to try to combine the models that perform best. The group (or "ensemble") will often perform better than the best individual model (just like Random Forests perform better than the best individual Decision Trees they rely on), especially if the individual models make very different types of errors.


**Suggestions are always welcome, thank you!**
