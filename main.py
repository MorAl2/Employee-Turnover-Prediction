import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from FCClassifier import FullyConnectedClassifier


def main():
    data = pd.read_csv('datasetHr.csv')
    labels = data['left']
    data = data.drop(columns=['left'])
    data = data.drop(columns=['Emp_Id'])
    data = pd.concat([data, pd.get_dummies(data['Department'], prefix='Dep')], axis=1)
    data = data.drop(columns=['Department'])

    data = pd.concat([data, pd.get_dummies(data['salary'], prefix='salary')], axis=1)
    data = data.drop(columns=['salary'])

    data['last_evaluation'] = data['last_evaluation'].apply(lambda x: x.replace("%", "")).astype(float)
    data['satisfaction_level'] = data['satisfaction_level'].apply(lambda x: x.replace("%", "")).astype(float)

    corrMatt = data.corr()
    #print(corrMatt['left'])
    # mask = np.array(corrMatt)
    # mask[np.tril_indices_from(mask)] = False
    # # thalachh maximum heart rate achieved has the highest correlation with the output
    # plt.figure(figsize=(12, 20))
    # plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(corrMatt, linewidths=0.6, vmax=1.0, mask=mask,
    #             square=True, linecolor='white', annot=True)
    # plt.show()

    data = np.array(data)
    labels = np.array(labels)

    # Logistic Regression Classifier:
    split_num = 10
    # kf = KFold(n_splits=split_num, shuffle=True)
    # acc_sum = 0

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=1)

    lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, criterion="entropy"))
    lr.fit(X_train, y_train)
    rf = lr.steps[-1][1]

    feature_names = ['satisfaction_level', 'last_evaluation', 'number_project',
                     'average_montly_hours', 'time_spend_company', 'Work_accident',
                     'promotion_last_5years', 'Dep_IT', 'Dep_RandD', 'Dep_accounting',
                     'Dep_hr', 'Dep_management', 'Dep_marketing', 'Dep_product_mng',
                     'Dep_sales', 'Dep_support', 'Dep_technical', 'salary_high',
                     'salary_low', 'salary_medium']

    # print( rf.feature_importances_)

    print(
        sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),
               reverse=True))

    y_hat = lr.predict(X_test)

    # score
    acc = accuracy_score(y_test, y_hat)
    print(f'Acc: {acc}')

    # resuts
    cm = confusion_matrix(y_test, y_hat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=1)

    lr = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=2))
    lr.fit(X_train, y_train)
    y_hat = lr.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print(f'Acc: {acc}')

    # resuts
    cm = confusion_matrix(y_test, y_hat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    # for train_index, test_index in kf.split(data):
    #     X_train, X_test = data[train_index], data[test_index]
    #     y_train, y_test = labels[train_index], labels[test_index]
    #     lr = make_pipeline(StandardScaler(), LogisticRegression())
    #     lr.fit(X_train, y_train)
    #     y_hat = lr.predict(X_test)
    #     #print(confusion_matrix(y_test, y_hat))
    #     acc = accuracy_score(y_test, y_hat)
    #     acc_sum += acc
    # print(f'Avg Acc Score: {acc_sum/split_num}')

    # # FC Classifier:
    # fc = FullyConnectedClassifier()
    # fc = fc.float()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(fc.parameters(), lr=0.001, momentum=0.9)
    # data = torch.tensor(data)
    # labels = torch.tensor(labels)
    #
    # for epoch in range(10):
    #     for i, data in enumerate(zip(data, labels), 0):
    #         # get the inputs
    #         inputs, y = data
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         outputs = fc(inputs.float())
    #         print(y)
    #         print(outputs)
    #         # y = torch.Tensor(y.float())
    #         # loss = criterion(outputs, y)
    #         # loss.backward()
    #         # optimizer.step()
    #         # print("loss: "+str(loss.item()))


if __name__ == "__main__":
    main()
