import math


class Gausiaan_Naive_Bayes:
    def __init__(self, train_x, train_y):
        # storing the training variables and target
        self.train_x = train_x
        self.train_y = train_y
        # storing distinct classes in alphabetical order
        self.distinct_class_list = sorted(list(set(self.train_y)))
        # counting total number of features
        self.total_features = len((self.train_x[0]))
        # attribute to store the mean for the various classes present in the training target
        # example mean={"ClassA":[10,20,30],"ClassB":[20,40,60]}
        self.mean = {}
        # same as mean. Using to store the standard deviation of all classes
        self.sd = {}
        # posterior probability i.e P(classA)=frequency of class A/total samples in training set
        self.prosterior_probability = {}
        # frequency count for all classes
        self.count = {}
        self.__setData()
        # method to calculate posterioir probabiliy
        self.__calculate_prosterior_probability()
        # method to calculate mean
        self.__calculate_mean()
        # method to calculate standard deviation
        self.__calculate_sd()

    def __setData(self):
        # initializing the attributes defined in constructor with 0 value for all classes for all features
        for i in self.distinct_class_list:
            class_key = i
            self.mean[class_key] = [0 for i in range(0, self.total_features)]
            self.sd[class_key] = [0 for i in range(0, self.total_features)]
            self.prosterior_probability[class_key] = 0
            self.count[class_key] = 0
        # calculating the count of frequencies per class
        for i in self.train_y:
            self.count[i] = self.count[i] + 1

    def __calculate_prosterior_probability(self):
        total_rows = len(self.train_y)
        # probability = P(classA)/total samples
        for i in self.distinct_class_list:
            self.prosterior_probability[i] = self.count[i]/total_rows

    def __calculate_mean(self):
        # mean is calculated per class
        # generating the numerator for mean i.e sum of all valuesm per class per feature
        for row in range(0, len(self.train_x)):
            #  i gives the row number of the dataset
            for col in range(0, self.total_features):
                # j gives the column number
                # self.train_y[i] gives the class name and j is used to access the mean for a particular feature
                self.mean[self.train_y[row]][col] = self.mean[self.train_y[row]
                                                              ][col] + self.train_x[row][col]
        # generating denominator by dividing sum by total number of rows per class
        for col in range(0, self.total_features):
            for class_name in self.distinct_class_list:
                self.mean[class_name][col] = self.mean[class_name][col] / \
                    self.count[class_name]

    def __calculate_sd(self):
        # sd is given by sum(xi-mean)^2/Number of rows
        for row, target_value in enumerate(self.train_y):
            # gives column number
            for col in range(0, self.total_features):
                self.sd[target_value][col] = self.sd[target_value][col] + \
                    (self.train_x[row][col] -
                     self.mean[target_value][col])**2
        for class_name in self.distinct_class_list:
            for feature in range(0, self.total_features):
                self.sd[class_name][feature] = self.sd[class_name][feature] / \
                    self.count[class_name]

    def __calculate_probability(self, value, mean, sd):
        # calculate the probability that the test value will occur for a class using gaussian distribution
        return math.exp((((value-mean)**2)/sd)/-2)/(2*math.pi*sd)**0.5

    def __calculate_log(self, prob_list):
        new_log_list = []
        for i in prob_list:
            try:
                new_log_list.append(math.log(i))
            except:
                new_log_list.append(0)
        # print(new_log_list)
        return new_log_list

    def __probability_test_feature(self, test_x):
        # calculate proability per class per feature
        # Prob(class1|x) = prob(x|class1)*prob(class1)
        # prob(x|class1) = prob(x1|class1)*prob(x2|class1)*prob(x3|class1)
        prob_class_feat = {}
        # {"ClassA":[prob_feat_1,prob_feat_2,],"ClassB":[prob_feat_1,prob_feat_2]}
        for i in self.distinct_class_list:
            prob_class_feat[i] = []
        for class_name in self.distinct_class_list:
            for value in test_x:
                # for every value in test calculating the probability per feature
                feat_prob_list = []
                for feat in range(0, self.total_features):
                    feat_prob_list.append(self.__calculate_probability(
                        value[feat], self.mean[class_name][feat], self.sd[class_name][feat]))
                # assuming that each feature is independent summing the log all probabilities to one another
                # and to corresponding posterior prob of classs to be able to handle under flow conditions
                prob_class_feat[class_name].append(
                    (sum(self.__calculate_log(feat_prob_list))+math.log(self.prosterior_probability[class_name])))

        return prob_class_feat

    def predictions(self, test_x):
        # getting probability for all values of test_x
        overall_prob = self.__probability_test_feature(test_x)
        # print(overall_prob)
        prediction_list = []
        # for every row of test_x for every class checking the highest probability to predict class
        for row, value in enumerate(test_x):
            # log value of probability will never be this big
            max_prob = -10000000000000000000000000000000000
            map_prob_class_name = ''
            for class_name in self.distinct_class_list:
                if(overall_prob[class_name][row] > max_prob):
                    max_prob = overall_prob[class_name][row]
                    # getting the corresponding class name for maximum probability
                    map_prob_class_name = class_name
            prediction_list.append(map_prob_class_name)
        return prediction_list

    def accuracy_score(self, predictions, test_y):
        # calculating accuracy = 100*(total correct predictions/total number of predictions)
        score = 0
        for i, value in enumerate(predictions):
            if(value == test_y[i]):
                score = score+1
        return (score/len(predictions)*100)

    def confusion_matrix(self, predictions, test_y):
        # generating confusion matrix
        # row depicts actual value
        #  column depicts predicted value
        confusion_matrix_list = []
        for i in range(0, len(self.distinct_class_list)):
            confusion_matrix_list.append(
                [0 for j in range(0, len(self.distinct_class_list))])
        for j in range(0, len(predictions)):
            # if prediction matches actual value corresponding diagonal value will be filled
            if(predictions[j] == test_y[j]):
                index = self.distinct_class_list.index(predictions[j])
                confusion_matrix_list[index][index] += 1
            else:
                # row number from actual value
                index_row = self.distinct_class_list.index(test_y[j])
                # column number from predicted value
                index_column = self.distinct_class_list.index(predictions[j])
                confusion_matrix_list[index_row][index_column] += 1
        return confusion_matrix_list

    def precision(self, predictions, test_y):
        precision_list = []
        # generating confusion matrix
        confusion_matrix_list = self.confusion_matrix(predictions, test_y)
        # precision = tp/(tp+fp)
        # for multi classes we have a list based on class
        for col in range(0, len(self.distinct_class_list)):
            tp = 0
            fp = 0
            for row in range(0, len(self.distinct_class_list)):
                if(col == row):
                    tp += confusion_matrix_list[row][col]
                else:
                    fp += confusion_matrix_list[row][col]
            precision_list.append(tp/(tp+fp))
        return precision_list

    def recall(self, predictions, test_y):
        # similar to precision but now we go row wise to get value of fn
        recall_list = []
        confusion_matrix_list = self.confusion_matrix(predictions, test_y)
        for row in range(0, len(self.distinct_class_list)):
            tp = 0
            fn = 0
            for col in range(0, len(self.distinct_class_list)):
                if(col == row):
                    tp += confusion_matrix_list[row][col]
                else:
                    fn += confusion_matrix_list[row][col]
            recall_list.append(tp/(tp+fn))
        return recall_list
