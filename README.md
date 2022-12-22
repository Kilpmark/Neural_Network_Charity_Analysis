# Neural_Network_Charity_Analysis

## Overview

An organization regulaly gives to various charity and non-profit groups. To better target their funding, this organization has collected data on thousands of donations in order to determine if factors within the data can predict the effect of the donation.

The organization has asked that a machine learning model, specifically a small neural network, be created to analyize patterns in the data.


## Results

### Data Preprocessing

-   After examining the various categories, the "IS_SUCCESSFUL" variable was chosen as the target. Although the measure of success is unknow, its binary nature yielded a more simple model than a than a several valued target.
- After applying binning as needed and performing encoding the initial feature set was:

![initial_features](/assets/initial_features.png)

- For the initial model, all variables listed above were considered to be features to provide a baseline result for future feature reduction.


### Compiling, Training, and Evaluating the Model

- For consistency, each model used the following setup with two Dense layers applyping ReLU and an output layer applying the sigmoid function. Adam was used as the optimizer:

![basic_model](/assets/basic_model.png)

- It should be noted that this is not the optimal setup but allows for better comparison between different modifications to the training data. The initial model produced these results, short of the goal of 75% accuracy.

![initial_eval](/assets/first_eval.png)

Of many possible scenarios, three were chosen for investigation. First, the STATUS feature was dropped. Almost all of the values were 1. Additionally, the APPLICATION_TYPE feature was rebinned to produce fewer columns. This model produced the following results:

![first_revision](/assets/sec_eval.png)

Potentially a small improvement from the original. The second revision was a continuation from the reductions performed in the first. The ASK_AMT feature is the only feature that was not binary. Using the method of Median +/- 1.5*IQR, all rows with potential outliers were dropped as shown:

![IQR](/assets/IQR.png)

The resulting model yielded these results:

![second_revision](/assets/third_eval.png)

Though it still does note provide the disired level of accuracy, this does appear to be an improvement over the original.

The third revision continued where the second left off. A Random Forest Classifier was used to identify features by importance.

![rf_results](/assets/rf.png)

Any feature representing more than 5% of the importance was kept. The resulting model suggests that two many features were discarded during this process and that a lower threshold should be employed.

![third_revision](/assets/fourth_eval.png)

## Summary

Using the same model structure, the reduction of ASK_AMT appears of have had the greatest impact. For further investigation, other configurations of a neural network should be explored.

One way to achieve this is by creating a function that builds the model with a variablbe number of dense layers, a uniform number of neurons in all but the output layer, and allows more granular tuning of hyperparameters. In this case it may be advantageous to employ Nesterov Accelerated Gradient Decent as the optimizing function because it may converge to a better solution than regular Adam for a less complex problem such as this.

After the function is created and activation functions, kernal initializers, the optimizer, and reasonable defaults are chosen it can be made into a regressor object (allowing interaction with Scikit-learn) by creating a KerasClassifier instance.

This regressor can be used with a dictionary of potential parameters to perform grid or randomized search cross fold validation. Given enough CPU time and power, a solution providing 75% accuracy could be found.

A different approach would be to use a Random Forest Classifier as the model itsel, although the image above did not show required performance, further or different tuning of the data could potentially achieve this.