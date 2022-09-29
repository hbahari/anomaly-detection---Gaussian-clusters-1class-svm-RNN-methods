import pandas as pd
import numpy as np
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
# from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time #helper libraries
from keras.models import model_from_json
import sys

# import and check data format
df = pd.read_csv("realKnownCause/ambient_temperature_system_failure.csv")
df.info()
print(df.head(5))

# change date format and tempreture unit
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['value'] = (df['value'] - 32) * 5 / 9
# plt.plot(df['timestamp'], df['value'])
# plt.show()

df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)
df['categories'] = df['WeekDay'] * 2 + df['daylight']
print(df.head(10))
# creation of 4 distinct categories  (week end/day week & night/day)
a = df.loc[df['categories'] == 0, 'value']
b = df.loc[df['categories'] == 1, 'value']
c = df.loc[df['categories'] == 2, 'value']
d = df.loc[df['categories'] == 3, 'value']

fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(a)
b_heights, b_bins = np.histogram(b, bins=a_bins)
c_heights, c_bins = np.histogram(c, bins=a_bins)
d_heights, d_bins = np.histogram(d, bins=a_bins)

width = (a_bins[1] - a_bins[0]) / 6

ax.bar(a_bins[:-1], a_heights * 100 / a.count(), width=width, facecolor='blue', label='WeekEndNight')
ax.bar(b_bins[:-1] + width, (b_heights * 100 / b.count()), width=width, facecolor='green', label='WeekEndLight')
ax.bar(c_bins[:-1] + width * 2, (c_heights * 100 / c.count()), width=width, facecolor='red', label='WeekDayNight')
ax.bar(d_bins[:-1] + width * 3, (d_heights * 100 / d.count()), width=width, facecolor='black', label='WeekDayLight')

plt.legend()
plt.show()

# scale the data that we're using for modeling
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# reduce the dimentioality to 2 features
pca = PCA(n_components=2)
data = pca.fit_transform(data)
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
print(data.head())

# clustering method
# find the best number of centroids that would result in the highest score
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

# use 14 as the number of clusters
df['cluster'] = kmeans[14].predict(data)
df['principal_feature1'] = data[0]
df['principal_feature2'] = data[1]
df['cluster'].value_counts()

# plot the clusters with datapoints
fig, ax = plt.subplots()
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'pink', 4: 'black', 5: 'orange', 6: 'cyan', 7: 'yellow', 8: 'brown',
          9: 'purple', 10: 'white', 11: 'grey', 12: 'lightblue', 13: 'lightgreen', 14: 'darkgrey'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["cluster"].apply(lambda x: colors[x]))
plt.show()


# find the distance of each point and its centroid and find the anomalies
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i] - 1]
        distance._set_value(i, np.linalg.norm(Xa - Xb))
    return distance


outliers_fraction = 0.01
distance = getDistanceByPoint(data, kmeans[14])
number_of_outliers = int(outliers_fraction * len(distance))
threshold = distance.nlargest(number_of_outliers).min()
df['anomaly21'] = (distance >= threshold).astype(int)
fig, ax = plt.subplots()
colors = {0: 'blue', 1: 'red'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["anomaly21"].apply(lambda x: colors[x]))
plt.show()
# visualisation of anomaly throughout time (viz 1)
fig, ax = plt.subplots()
a = df.loc[df['anomaly21'] == 1, ['time_epoch', 'value']]  # anomaly
ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

# Categories and Gaussian method
# reuse the 4  categories that we made earlier a,b,c,d
df_class0 = df.loc[df['categories'] == 0, 'value']
df_class1 = df.loc[df['categories'] == 1, 'value']
df_class2 = df.loc[df['categories'] == 2, 'value']
df_class3 = df.loc[df['categories'] == 3, 'value']

# use envelope function for finding gaussian outliers
envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class0.values.reshape(-1, 1)
envelope.fit(X_train)
df_class0 = pd.DataFrame(df_class0)
df_class0['deviation'] = envelope.decision_function(X_train)
df_class0['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class1.values.reshape(-1, 1)
envelope.fit(X_train)
df_class1 = pd.DataFrame(df_class1)
df_class1['deviation'] = envelope.decision_function(X_train)
df_class1['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class2.values.reshape(-1, 1)
envelope.fit(X_train)
df_class2 = pd.DataFrame(df_class2)
df_class2['deviation'] = envelope.decision_function(X_train)
df_class2['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class3.values.reshape(-1, 1)
envelope.fit(X_train)
df_class3 = pd.DataFrame(df_class3)
df_class3['deviation'] = envelope.decision_function(X_train)
df_class3['anomaly'] = envelope.predict(X_train)

# plot the temperature repartition by categories with anomalies
a0 = df_class0.loc[df_class0['anomaly'] == 1, 'value']
b0 = df_class0.loc[df_class0['anomaly'] == -1, 'value']

a1 = df_class1.loc[df_class1['anomaly'] == 1, 'value']
b1 = df_class1.loc[df_class1['anomaly'] == -1, 'value']

a2 = df_class2.loc[df_class2['anomaly'] == 1, 'value']
b2 = df_class2.loc[df_class2['anomaly'] == -1, 'value']

a3 = df_class3.loc[df_class3['anomaly'] == 1, 'value']
b3 = df_class3.loc[df_class3['anomaly'] == -1, 'value']

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist([a0, b0], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[0, 1].hist([a1, b1], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[1, 0].hist([a2, b2], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[1, 1].hist([a3, b3], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[0, 0].set_title("WeekEndNight")
axs[0, 1].set_title("WeekEndLight")
axs[1, 0].set_title("WeekDayNight")
axs[1, 1].set_title("WeekDayLight")
plt.legend()
plt.show()

# add to the main df
df_class = pd.concat([df_class0, df_class1, df_class2, df_class3])
df['anomaly22'] = df_class['anomaly']
df['anomaly22'] = np.array(df['anomaly22'] == -1).astype(int)

# Markov chain method
# definition of the different state
x1 = (df['value'] <= 18).astype(int)
x2 = ((df['value'] > 18) & (df['value'] <= 21)).astype(int)
x3 = ((df['value'] > 21) & (df['value'] <= 24)).astype(int)
x4 = ((df['value'] > 24) & (df['value'] <= 27)).astype(int)
x5 = (df['value'] > 27).astype(int)
df_mm = x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5

# def getTransitionMatrix(df):
#     df = np.array(df)
#     model = msm.estimate_markov_model(df, 1)
#     return model.transition_matrix
#
#
# def markovAnomaly(df, windows_size, threshold):
#     transition_matrix = getTransitionMatrix(df)
#     real_threshold = threshold ** windows_size
#     df_anomaly = []
#     for j in range(0, len(df)):
#         if (j < windows_size):
#             df_anomaly.append(0)
#         else:
#             sequence = df[j - windows_size:j]
#             sequence = sequence.reset_index(drop=True)
#             df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
#     return df_anomaly
#
#
# df_anomaly = markovAnomaly(df_mm, 5, 0.20)
# df_anomaly = pd.Series(df_anomaly)
# print(df_anomaly.value_counts())
# df['anomaly24'] = df_anomaly
# fig, ax = plt.subplots()
#
# a = df.loc[df['anomaly24'] == 1, ('time_epoch', 'value')] #anomaly
#
# ax.plot(df['time_epoch'], df['value'], color='blue')
# ax.scatter(a['time_epoch'],a['value'], color='red')
# plt.show()

# isolation forest method
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# train isolation forest
model = IsolationForest(contamination=outliers_fraction)
model.fit(data)
# add the data to the main
df['anomaly25'] = pd.Series(model.predict(data))
df['anomaly25'] = df['anomaly25'].map({1: 0, -1: 1})
print(df['anomaly25'].value_counts())
fig, ax = plt.subplots()

a = df.loc[df['anomaly25'] == 1, ['time_epoch', 'value']]  # anomaly

ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

# one class svm
# Good for novelty detection (no anomalies in the train set). This algorithm performs well for multimodal data
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM
model = OneClassSVM(nu=0.95 * outliers_fraction)  # nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model.fit(data)
# add the data to the main
df['anomaly26'] = pd.Series(model.predict(data))
df['anomaly26'] = df['anomaly26'].map({1: 0, -1: 1})
print(df['anomaly26'].value_counts())
fig, ax = plt.subplots()

a = df.loc[df['anomaly26'] == 1, ['time_epoch', 'value']]  # anomaly
ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

# RNN method
# RNN learn to recognize sequence in the data and then make prediction based on the previous sequence.
# We consider an anomaly when the next data points are distant from RNN prediction. Aggregation, size of
# sequence and size of prediction for anomaly are important parameters to have relevant detection.
data_n = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data_n)
data_n = pd.DataFrame(np_scaled)

# important parameters and train/test size
prediction_time = 1
testdatasize = 1000
unroll_length = 50
testdatacut = testdatasize + unroll_length  + 1

#train data
x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ][0].as_matrix()

# test data
x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ][0].as_matrix()

#unroll: create sequence of 50 previous data points for each data points
def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

# adapt the datasets for the sequence data shape
x_train = unroll(x_train,unroll_length)
x_test  = unroll(x_test,unroll_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]

# see the shape
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

model = Sequential()

model.add(LSTM(
    input_dim=x_train.shape[-1],
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    units=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : {}'.format(time.time() - start))


model.fit(
    x_train,
    y_train,
    batch_size=3028,
    nb_epoch=30,
    validation_split=0.1)

loaded_model = model
diff=[]
ratio=[]
p = loaded_model.predict(x_test)
# predictions = lstm.predict_sequences_multiple(loaded_model, x_test, 50, 50)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))

fig, axs = plt.subplots()
axs.plot(p,color='red', label='prediction')
axs.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()
