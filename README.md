# COMP219 - 2018 - First CA Assignment
Document of implementation and evaluation of the result

## Loading data
The dataset Adult Data Set is selected in UCI Machine Learning Repository.
It is a widely used dataset to predict whether income exceeds $50K/year according to provided census data.
Several steps to input the data:
1. Import pandas and use the function read_csv to input the dataset
file "adult.data.txt".
2. Label the features of the file with Age, Workclass, fnlwgt, Education,
Education_Num, Martial_Status, Occupation, Relationship, Race, Sex, Capital_Gain, Capital_Loss, Hours_per_week, Country, Target.
3. Replace String data to numbers. From example, there are two strings in sex
label, Female and Male, which should be replaced by 0 and 1.
4. Replace missing NaN values with rational numbers by fillna function.
5. Select list of features and targets by:
X = data[features].values
y = data["target"]
6. Display numbers of entries by:
print('numbers of data entries: ',X.shape[0])

## Training data
First considered model is trained by logistic regression. (model 1-3 in code)
At the beginning, the basic regression is trained by this code:
```
lr3 = LogisticRegression(C = 1000.0, random_state = 0)
lr3.fit(X_train_std, y_train)
```

However, the result shows the Accuracy will be lower if we choose X_train_std rather than X_train which is conducted by transformation manually.
After that, there is an optimized algorithm by logistic regression:
```
def logistic_regression():
regression = LogisticRegression()
grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
#try to get best penalty from l1, l2 and C from 0.01, 0.1, 1, 10, 100, 1000
lr = GridSearchCV(regression, param_grid=grid, scoring='accuracy') lr.fit(X_train, y_train)
return lr.best_estimator_
```
The function chooses the best estimator from both of the penalty l1 and l2 and various C, then generate a grid to choose the best one.
Similarly, other models are trained by k-nearest neighbors. (model 3 in code)

After that, it will print out the misclassified samples and Accuracy to validate the successful training, a predict result of one sample will also display.

## Model evaluation
As for the evaluation of the models, since algorithm of logistic regression preforms well in GridSearchCV function, it is chosen to compare with traditional logistic regression and k-nearest neighbor algorithm.
Firstly, three tables are generated to show the result of models.

| |precision|recall|f1-score|support|
|--| -----:| :----: | :----:|:----:|
|Under 50k|0.87|0.93|0.90|7407|
|Over 50k|0.72|0.58|0.64|2362|
|avg / total|0.84|0.85 |0.84 | 9769|
 <center>Table 1: Classification result of linear regression with GridSearchCV function</center>

 
