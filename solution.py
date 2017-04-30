# coding: utf8
from __future__ import division
from pandas import read_csv, DataFrame, Series
import re
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score

#Function to clean the data from the list, stemm the words and return the list of cleaned strings
def get_words(lst, word_max):
    num_lines = len(lst)
    meaningful_words = []
    stemmed_words = ""
    stops_2 = [u'прод', u'прода', u'продаж', u'цен', u'стоим', u'хорош', u'отличн', u'качествен', u'нов', u'лучш', u'состоян', u'дешев', u'красн', u'желт', u'зелен', u'черн', u'бел', u'оранжев', u'фиолетов' ]
    i = 0
    stemmer = SnowballStemmer("russian")
    stops = set(stopwords.words("russian"))
    for title in lst:
        if( (i+1) % 50000 == 0 ):
            print "     processing line %d of %d" % (i+1, num_lines)
        letters_only = re.sub(u'[^А-Яа-яa-zA-Z0-9]',' ', title)
        lower_case = letters_only.lower()
        words = lower_case.split()
        words = words[:word_max]
        for w in words:
            w = stemmer.stem(w)
            if not w.strip() in stops and w.strip() not in stops_2:
                stemmed_words = stemmed_words+" "+w
        meaningful_words.append(stemmed_words.lstrip())
        stemmed_words = ""
        i = i+1
    return meaningful_words

#MAIN PARAMETERS
train_set_size = 150000 #size of training data
test_set_size = 243167 #size of test data
word_max = 200 #get_words maximum number of words analyzed in a string
features_max = 600 #TfidfVectorizer 250
estimators_max = 100 #RandomForestClassifier 100
accr = int(0.25*train_set_size)

# read train data
print "Reading training data..."
train = read_csv(r'data\train.csv', encoding = 'UTF8')
train_set_size = min(train_set_size, len(train['title'])-accr)
lst_train_title = train[:train_set_size+accr]['title']
lst_train_description = train[:train_set_size+accr]['description']
lst_train_price = train[:train_set_size+accr]['price']
lst_train_price = lst_train_price.reshape(len(lst_train_price),1)

# clean and parse train data. If we have the cleaned file, skip this phase
try:
    clean_data = read_csv(r'cleaned_data.csv', encoding = 'UTF8')
    print "reading clean_data.csv"
    clean_train_titles = clean_data['Clean titles']
    clean_train_description = clean_data['Clean description']
except:
    print "Cleaning training data and creating a TITLE list..."
    clean_train_titles = get_words(lst_train_title, word_max)
    print "Cleaning training data and creating a DESCRIPTION list..."
    clean_train_description = get_words(lst_train_description, word_max)
#    output = DataFrame( data={"item_id":train[:train_set_size+accr]["item_id"], "Clean titles":clean_train_titles, "Clean description":clean_train_description} )
#    output.to_csv( "cleaned_data.csv", index=False, quoting=3, encoding = 'UTF8')

# create tfidf vectorizer for bag of words
vectorizer_t = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = features_max)
vectorizer_d = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = features_max)

#Get a bag of words for the train set, and convert to a numpy array
train_title_features = vectorizer_t.fit_transform(clean_train_titles[:train_set_size])
train_title_features = train_title_features.toarray()
train_description_features = vectorizer_d.fit_transform(clean_train_description[:train_set_size])
train_description_features = train_description_features.toarray()

#Training on TITLE, DESCRIPION and PRICE
print "Training on TITLE, DESCRIPION and PRICE"
forest_t = RandomForestClassifier(n_estimators = estimators_max)
forest_p = RandomForestClassifier(n_estimators = estimators_max)
forest_d = RandomForestClassifier(n_estimators = estimators_max)

forest_t.fit(train_title_features, train[:train_set_size]['category_id'])
forest_d.fit(train_description_features, train[:train_set_size]['category_id'])
forest_p.fit(lst_train_price[:train_set_size], train[:train_set_size]['category_id'])

# calculate accuracy on the subset of the train dataset
try:
    cat_hierarch = read_csv(r'cat_hierarch.csv', encoding = 'UTF8')
    t_title_features = vectorizer_t.transform(clean_train_titles[train_set_size:train_set_size+accr])
    t_description_features = vectorizer_d.transform(clean_train_titles[train_set_size:train_set_size+accr])
    accr_pred_t = forest_t.predict(t_title_features)
    accr_pred_d = forest_d.predict(t_description_features)
    accr_pred_p = forest_p.predict(lst_train_price[train_set_size:train_set_size+accr])
    accr_result = []
    for i in range(len(accr_pred_t)):
        if accr_pred_d[i] == accr_pred_p[i]: accr_result.append(accr_pred_p[i])
        else: accr_result.append(accr_pred_t[i])
    #printing accuraxy for different lefels 1:3
    res_cat_1 = 0
    res_cat_2 = 0
    res_cat_3 = 0
    for i, j in zip(xrange(len(accr_result)), xrange(train_set_size,train_set_size+accr)):
        if int(cat_hierarch['category_1'][accr_result[i]]) == int(cat_hierarch['category_1'][train['category_id'][j]]): res_cat_1 = res_cat_1 + 1
        if int(cat_hierarch['category_2'][accr_result[i]]) == int(cat_hierarch['category_2'][train['category_id'][j]]): res_cat_2 = res_cat_2 + 1
        if int(cat_hierarch['category_3'][accr_result[i]]) == int(cat_hierarch['category_3'][train['category_id'][j]]): res_cat_3 = res_cat_3 + 1
    print "Accuracy at hierarchy level 1: ", '{0:.3f}' .format(res_cat_1/len(accr_result))
    print "Accuracy at hierarchy level 2: ", '{0:.3f}' .format(res_cat_2/len(accr_result))
    print "Accuracy at hierarchy level 3: ", '{0:.3f}' .format(res_cat_3/len(accr_result))
except:
    print "Put cat_hierarch.csv file in the same directory to get the accuracy on different levels of hierarchy."

# reading test data
print "Reading testing data..."
test = read_csv(r'data\test.csv', encoding = 'UTF8')
test_set_size = min(test_set_size, len(test['title']))
lst_test_title = test[0:test_set_size]['title']
lst_test_description = test[0:test_set_size]['description']
lst_test_price = test[0:test_set_size]['price']
lst_test_price = lst_test_price.reshape(len(lst_test_price),1)

# clean and parse test data
print "Cleaning testing data and creating a TITLE list..."
clean_test_titles = get_words(lst_test_title, word_max)
print "Cleaning testing data and creating a DESCRIPTION list..."
clean_test_description = get_words(lst_test_description, word_max)

# get a bag of words for the test set, and convert to a numpy array
test_title_features = vectorizer_t.transform(clean_test_titles)
test_title_features = test_title_features.toarray()
test_description_features = vectorizer_d.transform(clean_test_description)
test_description_features = test_description_features.toarray()

# use the random forest to make category_id label predictions
print "Predicting on tesing data..."
result_t = forest_t.predict(test_title_features)
result_d = forest_d.predict(test_description_features)
result_p = forest_p.predict(lst_test_price)

# use a voting approach to class determination
result = []
for i in range(len(result_t)):
    if result_d[i] == result_p[i]: result.append(result_p[i])
    else: result.append(result_t[i])

# copy the results to a pandas dataframe with an "item_id" column and a "category_id" column
print "Saving results to result.csv..."
output = DataFrame( data={"item_id":test[0:test_set_size]["item_id"], "category_id":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "result.csv", index=False, quoting=3)