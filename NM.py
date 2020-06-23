import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split  # dividing data into groups
from operator import itemgetter  # get each rows from pandas data type
from sklearn.metrics import confusion_matrix
import math

"""
    Class discribing dataset for one class
"""

class Class:
    name = 0
    feature_list=[]
    feture_array = []
    covariance_array = []

    def __init__(self):
        self.name = 0
        self.feature_list=[]
        self.covariance_array = []
        self.feture_array = []

    
# select from excel given classes and features 
def _prepare_data(path_to_csv,class_list,feature_list):
        """import data"""
        df = pd.read_csv(path_to_csv, header=None)
        """choose class"""
        k = df[(df[0].isin(class_list))]
        """return class with specific features"""
        return k[feature_list]
# set wich classes and features will be computed   
data = _prepare_data(path_to_csv="leaf.csv",class_list = [9, 10, 11, 13, 24,25,27,29],
                                            feature_list = [0,3,4,7,8,9,12])
# split dataset into train and test 
train, test = train_test_split(
            data.copy(), test_size=0.2, random_state=111)

test = np.array(test)  #cast test to np.array 


# select class names
class_names_list = list(dict.fromkeys(data[0].to_list()))

# create objects for class exists in collection
# example
# class.name = 3
# class.feature_list[1, 1, 1],
            #     [1, 2, 1],
            #     [1, 3, 2],
            #     [1, 4, 3],
            #     [2, 4, 2],
            #     [2, 3, 3],
            #     [2, 2, 1],
            #     [1, 4, 2]
# ])
# print(data)



class_object_list = []
i = 0
for c in class_names_list:
    class_object_list.append(Class())
    class_object_list[i].name = c
    i+=1
for z, test_item in train.iterrows():

#    get only features from list, element zero is class name 
   buff_list = test_item[1:]
#    find class and append feature list
   for obj in class_object_list:
        test1 = int(test_item[0])
        if test1 == obj.name:
            obj.feature_list.append(buff_list)

  
# cast to numpy.array, compute covariance and cast list to numpy.array
# print(class_object_list[0].feature_list)
# collect all prepared data in lists(from all class)
all_cov_list = []
all_feature_array_list = []
all_class_names_list = []



for T in class_object_list: 
    T.feature_array= np.array(T.feature_list)        # cast list to numpy.array
    T.covariance_array = np.cov(T.feature_array.T)   # compute covvariance
    all_cov_list.append(T.covariance_array) 
    all_feature_array_list.append(T.feature_array)
    all_class_names_list.append(T.name)

# mahalanobis
predicted_class_list = []  # list of predicted class 
true_class_list = []       # list of true class name 
positive_predict_counter = 0
for point in test:
    distances_list = []
    dist = []
    for gr_cov, gr, gr_name in zip(all_cov_list, all_feature_array_list, all_class_names_list):
        gr_mean = np.mean(gr, axis=0)
        gr_cov_inv = np.linalg.inv(gr_cov)
        distance_buff = distance.mahalanobis(point[1:], gr_mean, gr_cov_inv)
        if math.isnan(distance_buff): 
            raise AttributeError
        dist += [distance_buff]
        distances_list.append([dist[-1],gr_name,point[0]])
       
        # sort list by distance
    nearest_mean = sorted(
        distances_list, key=itemgetter(0))
    # print(nearest_mean)    
    predicted_class_list.append(nearest_mean[0][1])   
    true_class_list.append(nearest_mean[0][2])
    if nearest_mean[0][1] == nearest_mean[0][2]:
        positive_predict_counter += 1

# euklides
e_predicted_class_list = []  # list of predicted class 
e_true_class_list = []       # list of true class name 
e_positive_predict_counter = 0
for point in test:
    distances_list = []
    dist = []
    for o in class_object_list:
        gr_mean = np.mean(o.feature_list, axis=0)
        distance_buff = distance.euclidean(point[1:], gr_mean)
        if math.isnan(distance_buff): 
            raise AttributeError
        dist += [distance_buff]
        distances_list.append([dist[-1],o.name,point[0]])
    nearest_mean = sorted(
       distances_list, key=itemgetter(0))   
    # print(nearest_mean)  
    e_predicted_class_list.append(nearest_mean[0][1])   
    e_true_class_list.append(nearest_mean[0][2])
    if nearest_mean[0][1] == nearest_mean[0][2]:
        e_positive_predict_counter += 1
   
   



print("Mahalanobis:")
print("Tablica pomyłek:")
print(confusion_matrix(true_class_list, predicted_class_list))
print("Wielkosc testowego zbioru: " + str(len(test)))
result_in_percent = (positive_predict_counter / len(test)) * 100
print("Skuteczność: ")
print(result_in_percent)
print()

print("Euklides:")
print("Tablica pomyłek:")
print(confusion_matrix(e_true_class_list, e_predicted_class_list))
print("Wielkosc testowego zbioru: " + str(len(test)))
result_in_percent = (e_positive_predict_counter / len(test)) * 100
print("Skuteczność: ")
print(result_in_percent)


 



