import pandas as pd  # analysis of tabular/json data
from operator import itemgetter  # get each rows from pandas data type
from numpy import math  # root
from sklearn.model_selection import train_test_split  # dividing data into groups
from statistics import mode
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix




class Knn:
    """
    Implementation of knn algorithm

    ...

    Attributes
    ----------
    k = 3  : int
        k parameter in algorithm
    feature_first : int
        first leaf feature
    feature_second : int
        second leaf feature
    first_class : float
        leaf class from instruction
    second_class : float
        leaf class from instruction
    new_types_list : list
        list of list that contain: distance between test and train point, class of predict class
    true_counter : int
        count positive predictions

    Methods
    -------
    prepare_data(sound=None)
        load and split data
    calculate_euclides_distance(sound=None)
        calculate distance between points
    prepare_data(sound=None)
        calculate distance between points
    calculate_euclides_distance(sound=None)
        calculate distance between points
    prepare_data(sound=None)
        calculate distance between points
    calculate_euclides_distance(sound=None)
        calculate distance between points
    """

    # k = 2
    # feature_list = [0, 1, 2]
    # class_list = [3,5]
    nn_list = []

    # feature_first = 2
    # feature_second = 6
    # first_class = 3.0
    # second_class = 5.0
    new_types_list = []
    true_counter = 0
    path_to_data_set = ""
    # x = 8
    # y = 10

    def __init__(self, k=4, feature_list = [0, 1, 2], class_list = [3, 5], path_to_data_set="leaf.csv"):
        self.k = k
        self.feature_list = feature_list
        self.class_list = class_list
        self.path_to_data_set = path_to_data_set

    def _prepare_data(self, path_to_csv):
        """import data"""
        df = pd.read_csv(path_to_csv, header=None)

        """choose class 3 i 5"""
        # return df[(df[0] == 3) | (df[0] == 5)]
        """choose class"""
        k = df[(df[0].isin(self.class_list))]
        """return class with specific features"""
        return k[self.feature_list]

    def _calculate_euclides_distance(self, traning_element, test_element, feature_first, feature_second):
        return math.sqrt((traning_element[feature_first] - test_element[feature_second]) ** 2 + (
            traning_element[feature_first] - test_element[feature_second]) ** 2)

    def _ndistance(self, traning_element, test_element):
        total = 0
        # print(traning_element)
        for i in range(len(traning_element)-1):
            diff = test_element[i+1] - traning_element[i+1]
            total += diff * diff

            
        return math.sqrt(total)

    def _get_k_nearest_neighbours(self, k_nearest_neighbours, first_class, k):
        counter_type_first = 0
        counter_type_second = 0
        for i in range(k):
            nn_leaf_class = k_nearest_neighbours[0]
            if nn_leaf_class[0] == first_class:
                counter_type_first += 1
            else:
                counter_type_second += 1
        return [counter_type_first, counter_type_second]

    def _predict_class_of_test_element(self, first_class, second_class, k_nearest_neighbours, k):
        counter_list = self._get_k_nearest_neighbours(
            k_nearest_neighbours, first_class, k)
        counter_type_first = counter_list[0]
        counter_type_second = counter_list[1]
        if counter_type_second > counter_type_first:
            checked_case_class = second_class
        else:
            checked_case_class = first_class
        return checked_case_class

    def check_algorithm_efficiency(self):
        data = self._prepare_data(self.path_to_data_set)
        y_true = []
        y_pred = []

        train2, test2 = train_test_split(
            data.copy(), test_size=0.2, random_state=111)
        # print(len(train2))    
        # print(train2)    
        if self.k >  len(train2):
            raise AttributeError  
        """
              Split data into two groups names test and train
        
               Parameters
               ----------
               test_size : float
                   train size in percent  
               random_state : random parameter(seed)     
        """
        for z, test_item in test2.iterrows():
            """loop iterating in test elements"""

            self.nn_list = []
            self.new_types_list = []

            for o, train_item in train2.iterrows():
                """loop iterating in training elements"""

                # dist = self._calculate_euclides_distance(traning_element=train_item, test_element=test_item,
                #                                          feature_first=self.feature_first,
                #                                          feature_second=self.feature_second)
                dist = self._ndistance(traning_element=train_item.values.tolist(
                ), test_element=test_item.values.tolist())

                self.new_types_list.append([train_item.values[0], dist])

            k_nearest_neighbours = sorted(
                self.new_types_list, key=itemgetter(1))

            # checked_case_class = self._predict_class_of_test_element(
            #     self.first_class, self.second_class, k_nearest_neighbours, self.k)
            for i in range(self.k):
                pair = k_nearest_neighbours[i]
                self.nn_list.append(pair[0])

            c = Counter(self.nn_list)
            # print(self.nn_list)
            count_class_list = c.most_common(self.k)
            print(count_class_list)
            checked_case_class = count_class_list[0]
            print(checked_case_class[0])
            print(test_item[0])
            y_pred.append(checked_case_class[0])
            y_true.append(test_item[0])

            if checked_case_class[0] == test_item[0]:
                self.true_counter += 1
       
        # for z, test_item in test2.iterrows():
        #     # print(train)
        #     plt.plot(test_item[1],test_item[2],marker='s',color="red") 
        # for z, train_item in train2.iterrows():
        #     # print(train)
        #     if train_item[0] == 3:
        #         plt.plot(round(train_item[self.x],2),round(train_item[self.y],2), marker='*',color="green")     
        #     else:     
        #         # print(train_item)
        #         plt.plot(train_item[self.x],train_item[self.y],marker='$f$',color="green")
        # plt.scatter(test2[self.x],test2[self.y])

        # for z, test_item in test2.iterrows():
        #     # print(train)
        #     if test_item[0] == 3:
        #         plt.plot(test_item[self.x],test_item[self.y],marker='*',color="red")     
        #     else:   
        #         plt.plot(test_item[self.x],test_item[self.y],marker='$f$',color="red")
        # plt.scatter(test2[6],test2[2])                  
        print("Tablica pomyłek:")
        print(confusion_matrix(y_true, y_pred))
        print("Wielkosc testowego zbioru: " + str(len(test2)))
        result_in_percent = (self.true_counter / len(test2)) * 100
        print("Skuteczność: ")
        print(result_in_percent)
        
        

        
        # plt.show() 



# usage - feature_list must include 0, because it contain name of class
knn = Knn()
knn.check_algorithm_efficiency()
