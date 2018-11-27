import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
import numpy as np
original_data = pd.read_csv(
    "adult.data.txt",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education_Num", "Martial_Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
        "Hours_per_week", "Country", "Target"],
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
# input the data and read the data skipping any spaces before/after the commas and mark the values ‘?’ as missing data points

original_data['sex'] = original_data.Sex.map({'Female': 0, 'Male': 1})
original_data['race'] = original_data.Race.map({'White': 0, 'Black': 1,
                                                'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3})
original_data['marital'] = original_data.Martial_Status.map({'Widowed': 0, 'Divorced': 1, 'Separated': 2,
                                                             'Never-married': 3, 'Married-civ-spouse': 4,
                                                             'Married-spouse-absent': 5, 'Married-AF-spouse': 6})
original_data['rel'] = original_data.Relationship.map({'Not-in-family': 0, 'Unmarried': 0,
                                                       'Own-child': 0, 'Other-relative': 0,
                                                       'Husband': 1, 'Wife': 1})
original_data['work'] = original_data.Workclass.map({'?': 0, 'Private': 1, 'State-gov': 2, 'Federal-gov': 3,
                                                     'Self-emp-not-inc': 4, 'Self-emp-inc': 5, 'Local-gov': 6,
                                                     'Without-pay': 7, 'Never-worked': 8})

original_data['target'] = original_data.Target.map({'<=50K': 0, '>50K': 1})
cols = ['Age', 'sex', 'race', 'Education_Num', 'work',
        'marital', 'rel', 'Hours_per_week', 'Capital_Gain', 'Capital_Loss',
        'fnlwgt', 'target']
# label string to number in order to calculate
data = original_data[cols].fillna(-9999)
# fill  rational numbers into missing NaN values

features = ['Age', 'sex', 'race', 'Education_Num', 'work',
            'marital', 'rel', 'Hours_per_week', 'Capital_Gain',
            'Capital_Loss', 'fnlwgt']
X = data[features].values
# select features
y = data["target"]
# select target
print('numbers of data entries: ',X.shape[0])
print('numbers of data features: ',X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
#split dataset into two pieces

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#calculate sample mean and standard deviation


def logistic_regression():

    regression = LogisticRegression()
    grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
#try to get best penalty from l1, l2 and C from 0.01, 0.1, 1, 10, 100, 1000
    lr = GridSearchCV(regression, param_grid=grid, scoring='accuracy')
    lr.fit(X_train, y_train)

    return lr.best_estimator_


def knearestneighbors():
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn
#k-k-nearestneighbors algorithm 
    
def trainingResult(model):
    
    y_pred = model.predict(X_test)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print("predict: ",model.predict([X_train[100,:]]))
    print("true: ",[y_train[100]])


def evaluation(model):
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred,
                                target_names=['Under 50k', 'Over 50k']))
    print('Training Set Accuracy Score: {:.2f}'.format(
        model.score(X_train, y_train)))
    print('Testing Set Accuracy Score: {:.2f}'.format(
        model.score(X_test, y_test)))
#print a series of information about the model


def confusion(model):

    y_pred = model.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    df = pd.DataFrame(confusion)
    plt.figure()
    sns.heatmap(df, annot=True)
    plt.title('Model Accuracy:{:.3f}'.format(model.score(X_test, y_test)))
#plot and display the confusion of the model

def precision_recall(model1, model2, model3):
    
    proba1 = model1.predict_proba(X_test)
    proba2 = model2.predict_proba(X_test)
    proba3 = model3.predict_proba(X_test)

    precision1, recall1, _ = precision_recall_curve(y_test, proba1[:,1])
    precision2, recall2, _ = precision_recall_curve(y_test, proba2[:,1])
    precision3, recall3, _ = precision_recall_curve(y_test, proba3[:,1])

    plt.figure()
    plt.plot(precision1, recall1, label='model1')
    plt.plot(precision2, recall2, label='model2')

    plt.plot(precision3, recall3, label='model3')   
    plt.legend(loc='upper right')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
#plot and display Precision-Recall curve

    
def Roc_curve(model1, model2, model3):
     
    proba1 = model1.predict_proba(X_test)
    proba2 = model2.predict_proba(X_test)
    proba3 = model3.predict_proba(X_test)
    fpr1, tpr1, _ = roc_curve(y_test, proba1[:, 1])
    fpr2, tpr2, _ = roc_curve(y_test, proba2[:, 1])
    fpr3, tpr3, _ = roc_curve(y_test, proba3[:, 1])
    plt.figure()
    plt.plot(fpr1, tpr1,label='model1')
    plt.plot(fpr2, tpr2,label='model2')
    plt.plot(fpr3, tpr3,label='model3')
    plt.legend(loc='upper right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve')
    plt.figure()
    name_list = ['model1','model2','model3']
    num_list = [auc(fpr1, tpr1),auc(fpr2, tpr2), auc(fpr3, tpr3)]
    plt.barh(range(len(num_list)), num_list,tick_label = name_list)
    plt.show()
#plot and display Roc curve and calculate auc of the model


def evaluation_bar(model1, model2, model3):
    fpr1, tpr1, _ = roc_curve(y_test, model1.predict_proba(X_test)[:, 1])
    fpr2, tpr2, _ = roc_curve(y_test, model2.predict_proba(X_test)[:, 1])
    fpr3, tpr3, _ = roc_curve(y_test, model3.predict_proba(X_test)[:, 1])
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    auc3 = auc(fpr3, tpr3)
    accuracy1 = accuracy_score(y_test, model1.predict(X_test))
    accuracy2 = accuracy_score(y_test, model2.predict(X_test))
    accuracy3 = accuracy_score(y_test, model3.predict(X_test))
    MSE1 = mean_squared_error(y_test, model1.predict(X_test))
    MSE2 = mean_squared_error(y_test, model2.predict(X_test))    
    MSE3 = mean_squared_error(y_test, model3.predict(X_test)) 
    precision1 = precision_score(y_test, model1.predict(X_test))
    precision2 = precision_score(y_test, model2.predict(X_test))
    precision3 = precision_score(y_test, model3.predict(X_test))
    
    name_list = ['AUC','Accuracy','MSE','precision']
    num_list1 = [auc1, accuracy1, MSE1, precision1]
    num_list2 = [auc2, accuracy2, MSE2, precision2]
    num_list3 = [auc3, accuracy3, MSE3, precision3]
    x =np.arange(4) 
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, num_list1,  width=width, label='model1',tick_label = name_list)
    plt.bar(x + width, num_list2, width=width, label='model2')
    plt.bar(x + 2 * width, num_list3, width=width, label='model3')
    plt.legend(loc='lower left')
    plt.show()  
    
    
    
model1 = logistic_regression()
model2 = LogisticRegression(C = 1000.0, random_state = 0)
model2.fit(X_train, y_train)
model3 = knearestneighbors()
model3.fit(X_train, y_train)
trainingResult(model1)
trainingResult(model2)
trainingResult(model3)
evaluation(model1)
evaluation(model2)
evaluation(model3)

confusion(model1)
confusion(model2)
confusion(model3)
precision_recall(model1,model2,model3)
Roc_curve(model1,model2,model3)
evaluation_bar(model1, model2, model3)