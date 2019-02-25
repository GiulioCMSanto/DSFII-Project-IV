import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (train_test_split,
									 StratifiedShuffleSplit,
									 StratifiedKFold,
									 GridSearchCV)
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score,
							precision_score,
						    recall_score,
						    f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

def print_dataset_info(dataset):
	"""
	This Function display basic informations about the dataset.
	"""

	##General Information
	print("Number of People in the dictionary: {}"
      	  .format(len(dataset.keys())))
	print("Number of features for pearson: {}"
      	  .format(len(dataset["SKILLING JEFFREY K"].keys())))
	print("Number of POI: {}"
      	  .format(sum(1 for i in dataset if dataset[i]["poi"] == 1)))
	print("\n")
	print("Jeffrey Skilling Attributes: \n")
	print(dataset["SKILLING JEFFREY K"])
	print("\n")

	##Number of NaNs
	#I will first create an empty dictionary with the 21 features
	keys = []
	for key in dataset["SKILLING JEFFREY K"].keys():
		keys.append(key)
	nan_data = dict.fromkeys(keys,0)

	#Now, I will increment the number of NaN for each feature
	for key in dataset:
	    for element in dataset[key]:
	        if dataset[key][element] == 'NaN':
	            nan_data[element] = nan_data[element] + 1
	print("Number of NaN per Attribute: \n")
	print(nan_data)
	print("\n")

def detect_outliers(dataset, nan_bias):
	"""
		This function identifies people with more than "nan_bias" NaN attributes
		and also the origin of too high values of salary and bonus.

		Parameters:
			dataset: the dataframe being analyzed
			nan_bias: the minimum accepted number of NaNs attributes for an individual
	"""
	#We have 21 features. Lets verefy people with more than nan_bias features with NaN	

	number_of_nan_features = 0
	for key in dataset:
		number_of_nan_features = 0
		for element in dataset[key]:
			if dataset[key][element] == 'NaN':
				number_of_nan_features = number_of_nan_features + 1
		if number_of_nan_features > nan_bias:
			print(key)
			print("\n")
			print(dataset[key])
			print("\n")

	#Lets verify very large values of bonus and salary
	features = ["salary", "bonus"]
	data = featureFormat(dataset, features)
	max_salary = max(data[:,0])
	max_bonus = max(data[:,1])
	print("Max Bonus: {}".format(max_bonus))
	print("Max Salary: {}".format(max_salary))

	for key in dataset:
	    if dataset[key]['bonus'] == max_bonus or dataset[key]['salary'] == max_salary:
	        print("Name of person with those values: {}".format(key))

	#Lets print out these points
	for point in data:
	    salary = point[0]
	    bonus = point[1]
	    plt.scatter( salary, bonus )
	plt.xlabel("salary")
	plt.ylabel("bonus")
	plt.show()


def outlier_removal(dataset,outliers):
	"""
		Given a dataframe and a list of outliers key, this function
		will pop out the outliers
	"""
	for outlier in outliers:
		dataset.pop(outlier,0)

	return dataset


def computeFraction(poi_messages, all_messages): #Function given by Udacity
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
	"""


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.

    if poi_messages == 0 or all_messages == 0:
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages)

    return fraction


def include_fraction_metrics(dataset):
	"""
		This function creates two metrics and include them in the dataset:
			- fraction_from_poi
			- fraction_to_poi
		A dataset with those attributes is returned
	"""

	for name in dataset:

	    data_point = dataset[name]

	    from_poi_to_this_person = data_point["from_poi_to_this_person"]
	    to_messages = data_point["to_messages"]
	    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
	    data_point["fraction_from_poi"] = fraction_from_poi


	    from_this_person_to_poi = data_point["from_this_person_to_poi"]
	    from_messages = data_point["from_messages"]
	    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
	    data_point["fraction_to_poi"] = fraction_to_poi

	    dataset[name]["fraction_from_poi"] = data_point["fraction_from_poi"]
	    dataset[name]["fraction_to_poi"] = data_point["fraction_to_poi"]

	return dataset

def extract_features(dataset):
	"""
	This function extracts all the features from a dataset and return them in a list.
	Notice: the feature POI is alocated in the first spot of the list!
	"""
	features_list = []
	for key in dataset["SKILLING JEFFREY K"].keys():
	    features_list.append(key)

	#Lets add the POI feature in the first position and remove the email_address
	features_list.remove('poi')
	features_list.remove('email_address')
	features_list.insert(0, 'poi')

	return features_list

def remove_nan_values(dataset):
	"""
	This Function replace the NaN values by 0 and return a new dataset.
	Parameters:
		dataset: the working dataframe
	"""
	for key in dataset:
		number_of_nan_features = 0
		for element in dataset[key]:
			if dataset[key][element] == 'NaN':
				if element == 'email_address':
					dataset[key][element] = ''
				else:
					dataset[key][element] = 0
	return dataset	

def choose_features(dataset, features_list, number_of_features):
	"""This function will return the best features for the modeling. 
	First, the function take the first 'number_of_features' top features using SelectKBest.
	Then, all possible combinations are made between those fetures and each list is evaluated
	through cross-validation (StratifiedShuffleSplit) using Gaussian NB, Decision Tree and K-Neighbors.
	Finally, a total score is calculated as the sum of the averages recall and precision and
	the set of features with highest total score is selected as the finalist.

	Parameters:
		dataset: the working dataframe
		features_list: a list with all features of intereset
		number_of_features: the number of top features that will be returned

	"""
	data = featureFormat(dataset, features_list, remove_NaN = False, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	selector = SelectKBest(f_classif,k="all")
	selector.fit(features,labels)

	features_scores = []
	for i in range(len(selector.scores_)):
	    features_scores.append((features_list[i+1],selector.scores_[i]))


	top_features_list = sorted(features_scores, key=lambda tup: tup[1],reverse=True)[:number_of_features]

	print("\n Most Powerful Selected Features: {} \n".format(top_features_list))

	final_list = []
	for i in range(number_of_features):
		final_list.append(top_features_list[i][0])

	##NOTICE: "using some features like from_poi_to_this_person, from_this_person_to_poi,
	#shared_receipt_with_poi to engineer new features can potentially create a data leakage." - Udacity review
	#Therefore, I will actually not be using those features.

	if 'fraction_to_poi' in final_list:
		final_list.remove('fraction_to_poi')
	if 'fraction_from_poi' in final_list:
		final_list.remove('fraction_from_poi')


	##Lest try a different combinations of features between the top ones to chose best set	
	#List of all combinations of features possible	
	#Reference: https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
	combs = []

	for i in xrange(1, len(final_list)+1):
	    els = [list(x) for x in itertools.combinations(final_list, i)]
	    for j in range(len(els)):
	    	els[j].insert(0, 'poi') #Alway include 'poi' in the first spot of the list
	    combs.extend(els)

	#Lets take the average score in a cross-validation for each features combination 
	#The classifier used to choose the parameters was a Naive Bayes becase of its simplicity
	list_of_scores = []

	for combination in combs:

		if len(combination) > 4: #Avoid bias

			#Obtain the data for modeling
			data = featureFormat(dataset, combination, remove_NaN = False, sort_keys = True)
			labels, features = targetFeatureSplit(data)

			#Lets take the average f1-score over a StratifiedShuffleSplit for each combination
			sss = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=42)

			#Letes evaluate the recall and the precision for each algorithm
			recall_1 = 0
			recall_2 = 0
			recall_3 = 0
			precision_1 = 0
			precision_2 = 0
			precision_3 = 0
			total_score_1 = 0
			total_score_2 = 0
			total_score_3 = 0
			total_score = 0

			for train_index, test_index in sss.split(features, labels):
				features_train = [features[ii] for ii in train_index]
				features_test = [features[ii] for ii in test_index]
				labels_train = [labels[ii] for ii in train_index]
				labels_test = [labels[ii] for ii in test_index]

				clf_1 = DecisionTreeClassifier()
				clf_2 = GaussianNB()
				clf_3 = KNeighborsClassifier()

				scaler = MinMaxScaler()
				reescaled_features_train = scaler.fit_transform(features_train)
				reescaled_features_test = scaler.transform(features_test)

				clf_1 = clf_1.fit(features_train, labels_train)
				clf_2 = clf_2.fit(features_train, labels_train)
				clf_3 = clf_3.fit(reescaled_features_train, labels_train)
	    		
				recall_1 = recall_1 + recall_score(labels_test,clf_1.predict(features_test))
				precision_1 = precision_1 + precision_score(labels_test,clf_1.predict(features_test))

				recall_2 = recall_2 + recall_score(labels_test,clf_2.predict(features_test))
				precision_2 = precision_2 + precision_score(labels_test,clf_2.predict(features_test))

				recall_3 = recall_3 + recall_score(labels_test,clf_3.predict(reescaled_features_test))
				precision_3 = precision_3 + precision_score(labels_test,clf_3.predict(reescaled_features_test))

			#Lets take the average score for the current features set
			recall_1 = recall_1/20
			precision_1 = precision_1/20

			recall_2 = recall_2/20
			precision_2 = precision_2/20

			recall_3 = recall_3/20
			precision_3 = precision_3/20

			total_score_1 = recall_1 + precision_1 
			total_score_2 = recall_2 + precision_2
			total_score_3 = recall_3 + precision_3

			#The total score is the sum of all scores
			total_score = total_score_1 + total_score_2 + total_score_3

			list_of_scores.append((total_score,combination))

	#Take the combination with highest total score	
	top_score = sorted(list_of_scores, key=lambda tup: tup[0],reverse=True)[0]
	
	print(" \n Best Obtained Combination and its F1-score: \n")
	print(top_score)

	#Thats the final feature list!
	final_list = top_score[1]
    
	return final_list


def classifier_tuner(dataset, scale, scaler, features_list, parameters, classifier, scoring):
	"""
	This function takes a classifier and its set of parameters and print the best estimator
	obtained under a StratifiedShuffleSplit cross-validation with the scoring metric.

	Parameters:
		dataset: the working dataset
		scale: True for scaling the features, False otherwise
		scaler: the desired scalers (Ex: MinMaxScaler())
		features_list: the optimal features list
		parameters: the set of parameters that will be compared
		classifier: the desired classifier
		scoring: the scoring method used in the cross-validation
	"""

	data = featureFormat(dataset, features_list, remove_NaN = False, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

	if scale:
		reescaled_features = scaler.fit_transform(features)
		clf = GridSearchCV(classifier, parameters, cv=sss, scoring=scoring)
		clf = clf.fit(reescaled_features, labels)
		print(clf.best_estimator_)
	elif not scale:
		clf = GridSearchCV(classifier, parameters, cv=sss, scoring=scoring)
		clf = clf.fit(features, labels)
		print(clf.best_estimator_)

def main():

	###TASK 1
	# Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
    	
    #Informations About the dataset
	print_dataset_info(dataset = data_dict)

	#Outlier Detection
	detect_outliers(dataset = data_dict, nan_bias = 16)
	#From above, the following people had almost all the features with NaN values:
	#[WODRASKA JOHN, WHALEY DAVID A, CLINE KENNETH W, WROBEL BRUCE, SCRIMSHAW MATTHEW,
	#GILLIS JOHN, THE TRAVEL AGENCY IN THE PARK]

	###TASK 2
	#Lets remove TOTAL and recalculate the largest bonus and salary
	data_dict.pop("TOTAL", 0)
	detect_outliers(dataset = data_dict, nan_bias = 16)

	##Lets remove [WODRASKA JOHN, WHALEY DAVID A, CLINE KENNETH W, WROBEL BRUCE, SCRIMSHAW MATTHEW,
	#GILLIS JOHN, THE TRAVEL AGENCY IN THE PARK]
	outliers = ["WODRASKA JOHN", "WHALEY DAVID A",
				"CLINE KENNETH W", "WROBEL BRUCE",
				"SCRIMSHAW MATTHEW","GILLIS JOHN",
				"THE TRAVEL AGENCY IN THE PARK"]

	data_dict = outlier_removal(dataset = data_dict, outliers = outliers)

	data_dict = remove_nan_values(dataset = data_dict)

	###TASK 3
	##In the 'Feature Selection' class, we looked at from_poi_to_this_person and
	##from_this_person_to_poi metrics. We concluded that a better metric is the
	##proportion one. Therefore, I will consider "fraction_from_poi" and
	##"fraction_to_poi" as new features and add them to the dictionary.

	data_dict = include_fraction_metrics(data_dict)

	#Lets verify "SKILLING JEFFREY K" to see if the attributes were included
	print("\n Jeffrey Skilling Attributs Again: \n")
	print(data_dict["SKILLING JEFFREY K"])

	### Store to my_dataset for easy export below.
	my_dataset = data_dict

	features_list = extract_features(dataset = my_dataset)
	print(" \n All features: \n")
	print(features_list)

	#features_list = choose_features(my_dataset,features_list,10)
	#Chosen list after running the above commented code (it takes some time to execute)

	#Without engineered features
	features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus', 'deferred_income']
	#With engineered features
	#features_list = ['poi','exercised_stock_options', 'total_stock_value', 'fraction_to_poi', 'deferred_income', 'long_term_incentive']

	### TASK 4 and TASK 5
	print("\n my_dataset length: {} \n".format(len(my_dataset)))
	print("\n features length: {} \n".format(len(features_list)))

	scaler = MinMaxScaler()

	##Algorithm 1: KNeighbors
	classifier = KNeighborsClassifier()
	n_neighbors = [i+1 for i in range(30)]
	parameters = {'n_neighbors': n_neighbors}
	classifier_tuner(dataset = my_dataset, scale = True, scaler = scaler, features_list = features_list,
					 parameters = parameters, classifier = classifier, scoring = 'f1')

	#Best parameter without engineered features: n_neighbors = 1
	#Best parameter with engineered features: n_neighbors = 3
	clf = KNeighborsClassifier(n_neighbors = 1)
	print("\n")
	test_classifier(clf, my_dataset, features_list)

	##Algorithm 2: Decision Tree
	classifier = DecisionTreeClassifier()
	min_samples_split = [i+5 for i in range(35)]
	parameters = {'min_samples_split':min_samples_split}
	classifier_tuner(dataset = my_dataset, scale = False, scaler = scaler, features_list = features_list,
					 parameters = parameters, classifier = classifier, scoring = 'f1')

	#Best parameter without engineered features: min_samples_split=14
	#Best parameter with engineered features: min_samples_split=13
	clf = DecisionTreeClassifier(min_samples_split = 14)
	print("\n")
	test_classifier(clf, my_dataset, features_list)

	##Algorithm 3: Suport Vector Machines
	#classifier = SVC()
	#parameters = {'kernel':('linear', 'rbf'),
    #          'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6],
    #          'gamma':[1e-6, 1e-4, 1e-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1e1, 11,
    #          		   12, 13, 14, 15, 16, 17, 18, 19, 20, 1e2, 1e3, 1e4]}
	#classifier_tuner(dataset = my_dataset, scale = True, scaler = scaler, features_list = features_list,
	#				 parameters = parameters, classifier = classifier, scoring = 'recall')

	#Best parameter: gamma = 2, kernel = 'rbf', C = 1000  
	#svc = SVC(kernel='rbf', C=1000, gamma = 2)
	svc = SVC(kernel='rbf', C=1000.0, gamma = 1)
	clf = Pipeline(steps = [('scaler', scaler), ('svc',svc)])
	test_classifier(clf, my_dataset, features_list)

	##Algorithm 4: Gaussian Naive Bayes
	clf = GaussianNB()
	test_classifier(clf, my_dataset, features_list)

	###TASK 6
	#Chosen algorithm: Gaussian NB
	clf = GaussianNB()
	dump_classifier_and_data(clf, my_dataset, features_list)
	
if __name__ == "__main__":
    main()