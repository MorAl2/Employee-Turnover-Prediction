import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




def main():
    data = pd.read_csv('datasetHr.csv')
    labels = data['left']

    data = data.drop(columns=['Emp_Id'])
    data = pd.concat([data, pd.get_dummies(data['Department'], prefix='Dep')], axis=1)
    data = data.drop(columns=['Department'])

    data = pd.concat([data, pd.get_dummies(data['salary'], prefix='salary')], axis=1)
    data = data.drop(columns=['salary'])

    data['last_evaluation'] = data['last_evaluation'].apply(lambda x: x.replace("%", "")).astype(float)
    data['satisfaction_level'] = data['satisfaction_level'].apply(lambda x: x.replace("%", "")).astype(float)

    # plotting the correlation table
    corrMatt = data.corr()

    print((  dict(sorted(corrMatt['left'].items(), key=lambda item: item[1]))))

    # print(corrMatt['left'])
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False
    # thalachh maximum heart rate achieved has the highest correlation with the output
    plt.figure(figsize=(12, 20))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(corrMatt, linewidths=0.6, vmax=1.0, mask=mask,
                square=True, linecolor='white', annot=True)
    plt.show()

    # dropping the label column
    data = data.drop(columns=['left'])
    data = np.array(data)
    labels = np.array(labels)

    # Random Forest Classifier:
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=1)

    # MinMax Scaling and RandomForestClassifier
    lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, criterion="entropy"))
    lr.fit(X_train, y_train)
    # extracting the model from pipeline.
    rf = lr.steps[-1][1]

    feature_names = ['satisfaction_level', 'last_evaluation', 'number_project',
                     'average_montly_hours', 'time_spend_company', 'Work_accident',
                     'promotion_last_5years', 'Dep_IT', 'Dep_RandD', 'Dep_accounting',
                     'Dep_hr', 'Dep_management', 'Dep_marketing', 'Dep_product_mng',
                     'Dep_sales', 'Dep_support', 'Dep_technical', 'salary_high',
                     'salary_low', 'salary_medium']

    # inspecting the most influencing features on the turnover decision.
    print(
        sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),
               reverse=True))

    y_hat = lr.predict(X_test)

    # score
    acc = accuracy_score(y_test, y_hat)
    print(f'Acc: {acc}')

    # detailed score analasys
    cm = confusion_matrix(y_test, y_hat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    # K Neighbors Classifier:
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=1)
    lr = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=2))
    lr.fit(X_train, y_train)
    y_hat = lr.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print(f'Acc: {acc}')

    # detailed score analasys
    cm = confusion_matrix(y_test, y_hat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


if __name__ == "__main__":
    main()
