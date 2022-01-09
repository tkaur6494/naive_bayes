# libraries needed for code execution
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as overall_score_sk, confusion_matrix
from tabulate import tabulate
import naive_bayes_implementaion as nb
import copy
import math
import random


# reading the input data for validation of algorithm
inputData = pd.read_excel('Reduced_Features_for_TAI_project.xlsx')
# --------------------First Part-----------------------------------------------------#
print("First Part Executing - Comparing performance without any feature selection")
# Implementing Gaussian Naive Bayes without any feature selection
# and comparing performance

# dictionary to store various key metrics from scikit and personal implementation
comparison_key_metrics = {"Implementation": [
    "Personal", "Sci-kit"], "Accuracy": [], "Precision": [], "Recall": []}

# assumption all micros for one patient have the same label
# classifying based on micros
aggregated_data = inputData.drop(["Patient ID"], axis=1)

# dropping duplicate columns
aggregated_data = inputData.drop(
    ["original_firstorder_Kurtosis", "original_firstorder_Skewness"], axis=1)

# getting summary of columns left to identify if there are any columns with 0 data or if the data needs to be scaled
# display(aggregated_data.describe())

# although the data values vary but since we are using naive bayes scaling will not have much effect
#  because naive bayes works with probabilities

# print(aggregated_data.info())

# Splitting the data in train and test samples.
x_train_fs, x_test_fs, y_train, y_test = train_test_split(
    aggregated_data.loc[:, aggregated_data.columns != 'Label'], aggregated_data['Label'], test_size=0.3, random_state=1)

# Using model built from sratch
naive_bayes_implement = nb.Gausiaan_Naive_Bayes(
    x_train_fs.values.tolist(), y_train.values.tolist())

#  storing list of predictions
predictions_implement = naive_bayes_implement.predictions(
    x_test_fs.values.tolist())

# getting the accuracy score,precsion and recall of the model
nb_accuracy_score = naive_bayes_implement.accuracy_score(
    predictions_implement, y_test.values.tolist())

comparison_key_metrics["Accuracy"].append(nb_accuracy_score)

nb_precision = naive_bayes_implement.precision(
    predictions_implement, y_test.values.tolist())
comparison_key_metrics["Precision"].append(nb_precision)

nb_recall = naive_bayes_implement.recall(
    predictions_implement, y_test.values.tolist())
comparison_key_metrics["Recall"].append(nb_recall)

# implementing gaussian naive bayes from scikit
naive_bayes_sk = GaussianNB()
naive_bayes_sk.fit(x_train_fs, y_train)
predictions_sk = naive_bayes_sk.predict(x_test_fs)
# print(naive_bayes_sk.predict_proba(x_test_fs).tolist())

nb_accuracy_sk = accuracy_score(y_test, predictions_sk)*100
comparison_key_metrics["Accuracy"].append(nb_accuracy_sk)

nb_precision_sk = overall_score_sk(predictions_sk, y_test)[1]
comparison_key_metrics["Precision"].append(nb_precision_sk)

nb_recall_sk = overall_score_sk(predictions_sk, y_test)[2]
comparison_key_metrics["Recall"].append(nb_recall_sk)
print("----------------------------------------------------")
print(tabulate(comparison_key_metrics, floatfmt=".2f", headers=[
      "Type", "Accuracy", "Precision", "Recall"]))
print("----------------------------------------------------")

# ----------------------------------End of first part---------------------------------------------------------- #

#  -----------------------------------Second Part---------------------------------------------------------------------
#  Tuning Hyperparameter k for feature selection and comparing performance
print("Second Part Executing - Tuning hyperparameter k for feature selection and comparing accuracy and time")
# Function to compute accuracy and time taken for both models


def compare_performance_feature(kmin, kmax, gap):

    X = aggregated_data.loc[:, aggregated_data.columns != 'Label']
    Y = aggregated_data['Label']
    # for plotting chart to compare the performances of both implemenattions based on number of features
    k_value = []
    accuracy_personal = []
    accuracy_sk = []
    time_taken_personal = []
    time_taken_sk = []
    precision_personal = []
    precision_sk = []
    recall_personal = []
    recall_sk = []
    for k in range(kmin, kmax, gap):
        k_value.append(k)
        start_time = time.time()
        # using Mutual information since input data is numerical and output is categorical
        fs = SelectKBest(score_func=mutual_info_classif, k=k)
        fs.fit(X, Y)
        input_X = fs.transform(X)
        x_train_fs, x_test_fs, y_train, y_test = train_test_split(
            input_X, Y, test_size=0.3, random_state=1)
        x_train_fs = x_train_fs.tolist()
        x_test_fs = x_test_fs.tolist()

        # # implementing the gaussian naive bayes created

        naive_bayes_implement = nb.Gausiaan_Naive_Bayes(
            x_train_fs, y_train.values.tolist())

        #  storing list of predictions
        predictions_implement = naive_bayes_implement.predictions(x_test_fs)

        # getting the accuracy score of the model
        nb_accuracy_score = naive_bayes_implement.accuracy_score(
            predictions_implement, y_test.values.tolist())
        accuracy_personal.append(nb_accuracy_score)

        nb_precision = naive_bayes_implement.precision(
            predictions_implement, y_test.values.tolist())
        precision_personal.append(nb_precision)

        nb_recall = naive_bayes_implement.recall(
            predictions_implement, y_test.values.tolist())
        recall_personal.append(nb_recall)

        time_taken_personal.append(time.time()-start_time)
        start_time_sk = time.time()

        # implementing gaussian naive bayes from scikit
        naive_bayes_sk = GaussianNB()
        naive_bayes_sk.fit(x_train_fs, y_train)
        predictions_sk = naive_bayes_sk.predict(x_test_fs)

        nb_accuracy_sk = accuracy_score(y_test, predictions_sk)*100
        accuracy_sk.append(nb_accuracy_sk)

        nb_precision_sk = overall_score_sk(predictions_sk, y_test)[1]
        precision_sk.append(nb_precision_sk)

        nb_recall_sk = overall_score_sk(predictions_sk, y_test)[2]
        recall_sk.append(nb_recall_sk)
        time_taken_sk.append(time.time()-start_time_sk)
    return (k_value, accuracy_personal, accuracy_sk, time_taken_personal, time_taken_sk, precision_personal, precision_sk, recall_personal, recall_sk)


k_value, acc_personal, acc_sk, time_personal, time_sk, pre_personal, pre_sk, recall_presonal, recall_sk = compare_performance_feature(
    5, 80, 5)

df_acc = pd.DataFrame(
    {"Personal": acc_personal, "Sci-Kit": acc_sk}, index=k_value)
df_acc.plot(kind="bar")
plt.title("Feature Count VS Accuracy")
plt.xlabel("k-value")
plt.ylabel("Accuracy (%)")
plt.show()

df_time = pd.DataFrame(
    {"Personal": time_personal, "Sci-Kit": time_sk}, index=k_value)
df_time["Personal"].plot(kind="line", legend=True)
df_time["Sci-Kit"].plot(kind="line", secondary_y=True, legend=True)
plt.title("Feature Count VS Time Taken")
plt.xlabel("k-value")
plt.ylabel("Time Taken (sec)")
plt.show()

#----------------------------------End of Second Part-------------------#

#---------------------Third Part------------------------------------------#
# Comparing performance between train-test split and 10 fold cross validation#
print("----------------------------------------------------")
print("Third Part Executing - Comparing accuracy of Train Test Split with 10 fold cross validation")

# Implementing cross validation


def cross_validation(input_x, input_y, k_fold):
    accuracy_personal = []
    accuracy_sk = []
    # Performing feature selection on the input dataset since this process does not need to be repeated
    fs = SelectKBest(score_func=mutual_info_classif, k=15)
    fs.fit(input_x, input_y)
    input_x = fs.transform(input_x)
    input_x = input_x.tolist()

    for i in range(0, k_fold):
        # Creating copy of the dataset since one backup will be needed
        train_x = copy.deepcopy(input_x)
        train_y = copy.deepcopy(input_y)
        test_x = []
        test_y = []

        # Generating random indices for splitting the dataset into train and test
        # Dividing the dataset into k-fold parts
        for j in range(0, math.floor(len(train_x)/k_fold)):
            index = random.randrange(len(train_x))
            # Adding random values from input and output set to the testing set
            test_x.append(train_x[index])
            test_y.append(train_y[index])
            # Removing the testing values from training set
            train_x.pop(index)
            train_y.pop(index)

        # implementing the gaussian naive bayes created

        naive_bayes_implement = nb.Gausiaan_Naive_Bayes(
            train_x, train_y)

        #  storing list of predictions
        predictions_implement = naive_bayes_implement.predictions(
            test_x)

        # getting the accuracy score of the model
        nb_accuracy_score = naive_bayes_implement.accuracy_score(
            predictions_implement, test_y)
        accuracy_personal.append(nb_accuracy_score)

        # implementing gaussian naive bayes from scikit
        naive_bayes_sk = GaussianNB()
        naive_bayes_sk.fit(train_x, train_y)
        predictions_sk = naive_bayes_sk.predict(test_x)

        nb_accuracy_sk = accuracy_score(test_y, predictions_sk)*100
        accuracy_sk.append(nb_accuracy_sk)
    return (accuracy_personal, accuracy_sk)


X = aggregated_data.loc[:, aggregated_data.columns != 'Label'].values.tolist()
Y = aggregated_data['Label'].values.tolist()
acc_personal_cross_valid, acc_sk_cross_valid = cross_validation(X, Y, 10)
print("10 Fold Cross Validation Accuracy- ")
print("Personal - "+str(round(sum(acc_personal_cross_valid) /
                              len(acc_personal_cross_valid), 2)))
print("Sci-Kit - "+str(round(sum(acc_sk_cross_valid)/len(acc_sk_cross_valid), 2)))
print("------------------------------------------------------------")
print("Train-Test Split - with 15 features Accuracy")
print(round(df_acc.loc[15, :], 2))


# -------------Third Part End------------------------------#
