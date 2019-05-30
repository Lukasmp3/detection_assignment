#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


# Data files
normal_traffic_file_1 = 'normalTrafficTraining.txt'
normal_traffic_file_2 = 'normalTrafficTest.txt'
anomalous_traffic_file = 'anomalousTrafficTest.txt'


# In[3]:


# Load data and remove newlines characters
def load_data(file_name):
    file = open(file_name, 'r')
    contents = file.read().split('\n')
    file.close()
    return contents


# In[4]:


# Load normal and anomalous data into seperate lists
normal_traffic = load_data(normal_traffic_file_1) + load_data(normal_traffic_file_2)
anomalous_traffic = load_data(anomalous_traffic_file)


# In[5]:


# Create a list of http requests
def create_requests_list(traffic_list):
    traffic_requests = []
    for line_num in range(len(traffic_list)):
        line = traffic_list[line_num]
        
        if line.startswith('GET'):
            # mozna jeste odstranim HTTP/1.1 ci na zacatku localhost
            # Remove unnecessary white spaces and uppercase
            line = line.lower().replace(' ', '')
            traffic_requests.append(line)
            
        elif line.startswith('POST') or line.startswith('PUT'):
            request_str = line
            while not line.startswith('Content-Length:'):
                line_num += 1
                line = traffic_list[line_num]
            # Second line below 'Content-Length' may be relevant
            request_str = request_str + traffic_list[line_num + 2]
            request_str = request_str.lower().replace(' ', '')
            traffic_requests.append(request_str)
    return traffic_requests


# In[6]:


normal_traffic_requests = create_requests_list(normal_traffic)
anomalous_traffic_requests = create_requests_list(anomalous_traffic)


# In[7]:


# Class normal: 1
# Class anomalous: 0
labels_normal = [1] * len(normal_traffic_requests)
labels_anomalous = [0] * len(anomalous_traffic_requests)

requests_all = normal_traffic_requests + anomalous_traffic_requests
labels_all = labels_normal + labels_anomalous


# In[8]:


# Vectorization of text data
# Could adjust TfidfVectorizer paramater settings, 5,5 works best
vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (5, 5))
X = vectorizer.fit_transform(requests_all)
#print(vectorizer1.get_feature_names())
#print(X.shape)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, labels_all, test_size = 0.1, random_state = 42)


# In[10]:


# Linear SVM - should scale better to large numbers of samples
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score_test = accuracy_score(y_test, y_pred)
print("Score Linear SVM: ", score_test)


# In[11]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(confusion_matrix)


# In[15]:


# Example of classification of some new samples
sample_traffic_file = 'sample_traffic.txt'
sample_traffic = load_data(sample_traffic_file)
traffic_request = create_requests_list(sample_traffic)
X_sample = vectorizer.transform(traffic_request)
sample_pred = clf.predict(X_sample)
for result in sample_pred:
    if result == 0:
        print('anomalous')
    else:
        print('normal')


# In[ ]:




