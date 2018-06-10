# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['OMP_NUM_THREADS'] = '8'

# Features
IP = 'ip'
APP = 'app'
DEVICE = 'device'
OS = 'os'
CHANNEL = 'channel'
CLICK_TIME = 'click_time'
ATTRIBUTED_TIME = 'attributed_time'
IS_ATTRIBUTED = 'is_attributed'
CLICK_ID='click_id'

# New features related to time
DAY_OF_WEEK = 'day_of_week'
DAY_OF_YEAR = 'day_of_year'

def timeFeatures(df):
    # Make some new features with click_time column
    df[DAY_OF_WEEK] = pd.to_datetime(df[CLICK_TIME]).dt.dayofweek
    df[DAY_OF_YEAR] = pd.to_datetime(df[CLICK_TIME]).dt.dayofyear
    df.drop([CLICK_TIME], axis=1, inplace=True)
    return df
    
TRAIN_COLUMNS = [IP, APP, DEVICE, OS, CHANNEL, CLICK_TIME, ATTRIBUTED_TIME, IS_ATTRIBUTED]
TEST_COLUMNS = [IP, APP, DEVICE, OS, CHANNEL, CLICK_TIME, CLICK_ID]

dtypes = {
    IP : 'int32',
    APP : 'int16',
    DEVICE : 'int16',
    OS : 'int16',
    CHANNEL : 'int16',
    IS_ATTRIBUTED : 'int8',
    CLICK_ID : 'int32'
}

# Train set
train_set = pd.read_csv('../input//train.csv', 
                        skiprows = range(1, 123903891), 
                        nrows=61000000, 
                        usecols=TRAIN_COLUMNS, 
                        dtype=dtypes)
# Test set
test_set = pd.read_csv('../input/test.csv', 
                       usecols=TEST_COLUMNS, 
                       dtype=dtypes)
                       
# Checkin Cremer V stats
# Method for cheking Cramer V stat
import scipy.stats as ss

def get_cramers_stat(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    # min(confusion_matrix.shape)-1 : 자유도
    cramers_stat = np.sqrt(phi2 / (min(confusion_matrix.shape)-1))
    return cramers_stat
    
# Ip Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.ip, train_set.is_attributed)))
# App Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.app, train_set.is_attributed)))
# Device Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.device, train_set.is_attributed)))
# OS Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.os, train_set.is_attributed)))
# Channel Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.channel, train_set.is_attributed)))

# Split Y 
y = train_set[IS_ATTRIBUTED]
train_set.drop([IS_ATTRIBUTED], axis=1, inplace=True)

# Sub dataframe is for submission.
sub = pd.DataFrame()
sub[CLICK_ID] = test_set[CLICK_ID]

nrow_train = train_set.shape[0]

merge = pd.concat([train_set, test_set])

del train_set, test_set
gc.collect()

# New features using group by
CLICKS_BY_IP = 'clicks_by_ip'
CLICKS_BY_IP_APP = 'clicks_by_ip_app'

# Count the number of clicked channels by ip
ip_count = merge.groupby([IP])[CHANNEL].count().reset_index()\
    .rename(columns = {CHANNEL : CLICKS_BY_IP})
merge = pd.merge(merge, ip_count, on=IP, how='left', sort=False)
merge[CLICKS_BY_IP] = merge[CLICKS_BY_IP].astype('int16')

# IP가 특정 app을 얼마나 많은 장소에서 클릭했는가?
ip_app_count = merge.groupby(by=[IP, APP])[CHANNEL].count().reset_index()\
    .rename(columns=({CHANNEL: CLICKS_BY_IP_APP}))
merge = pd.merge(merge, ip_app_count, on=[IP, APP], how='left', sort=False)
merge[CLICKS_BY_IP_APP] = merge[CLICKS_BY_IP_APP].astype('int16')

# Drop columns not necessary
# IP?
merge.drop([ATTRIBUTED_TIME, CLICK_ID], axis=1, inplace=True)

# Adding new features
merge = timeFeatures(merge)

train_set = merge[:nrow_train]
test_set = merge[nrow_train:]

# Train set, y를 join해서, clicks_by_ip와의 correlation을 구해본다.
correlation_df = pd.concat([train_set[[CLICKS_BY_IP, CLICKS_BY_IP_APP]], y], axis=1)
correlation = correlation_df.corr()
sns.heatmap(correlation, cmap='viridis', annot=True, linewidth=3)

del correlation_df, correlation
gc.collect()

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

# Max type이 float 값이다.
(max_app
,max_channel
,max_device
,max_os
,max_clicks_by_ip
,max_clicks_by_ip_app
,max_day_of_week
,max_day_of_year) = merge.describe().loc['max'].astype('int64') + 1

del merge
gc.collect()

# Max 값은 float 값으로!
emb_dim = 50
dense_n = 1000
# Embedding
# APP
input_app = Input(shape=[1], name=APP)
emb_app = Embedding(max_app, emb_dim)(input_app)
# CHANNEL
input_channel = Input(shape=[1], name=CHANNEL)
emb_channel = Embedding(max_channel, emb_dim)(input_channel)
# DEVICE
input_device = Input(shape=[1], name=DEVICE)
emb_device = Embedding(max_device, emb_dim)(input_device)
# OS
input_os = Input(shape=[1], name=OS)
emb_os = Embedding(max_os, emb_dim)(input_os)
# DAY_OF_WEEK
input_dow = Input(shape=[1], name=DAY_OF_WEEK)
emb_dow = Embedding(max_day_of_week, emb_dim)(input_dow)
# DAY_OF_YEAR
input_doy = Input(shape=[1], name=DAY_OF_YEAR)
emb_doy = Embedding(max_day_of_year, emb_dim)(input_doy)
# Numeric - embedding?
# CLICKS_BY_IP
input_ip_cnt = Input(shape=[1], name=CLICKS_BY_IP)
emb_ip_cnt = Embedding(max_clicks_by_ip, emb_dim)(input_ip_cnt)
# CLICKS_BY_IP_APP
input_ip_app_cnt = Input(shape=[1], name=CLICKS_BY_IP_APP)
emb_ip_app_cnt = Embedding(max_clicks_by_ip_app, emb_dim)(input_ip_app_cnt)

cols = concatenate(
    [emb_app, emb_channel, emb_device, emb_os, emb_dow, emb_doy, emb_ip_cnt, emb_ip_app_cnt])

# Layer and model
drop_out = SpatialDropout1D(0.2)(cols)
x = Flatten()(drop_out)
x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
output_prob = Dense(1, activation='sigmoid')(x)
model = Model(
    inputs = [
        input_app,
        input_channel,
        input_device,
        input_os,
        input_dow,
        input_doy,
        input_ip_cnt,
        input_ip_app_cnt
    ],
    outputs=output_prob
)

batch_size = 20000
epochs = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_set) / batch_size) * epochs
learing_rate_init = 0.001
learning_rate_finish = 0.0001
learing_rate_decay = exp_decay(learing_rate_init, learning_rate_finish, steps)
optimizer = Adam(decay=learing_rate_decay)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# Train - using numpy array
# Input of keras : numpy array
train_set = {
    APP: np.array(train_set.app),
    CHANNEL: np.array(train_set.channel),
    DEVICE: np.array(train_set.device),
    OS: np.array(train_set.os),
    CLICKS_BY_IP: np.array(train_set.clicks_by_ip),
    CLICKS_BY_IP_APP: np.array(train_set.clicks_by_ip_app),
    DAY_OF_WEEK: np.array(train_set.day_of_week),
    DAY_OF_YEAR: np.array(train_set.day_of_year)
}

model.fit(train_set,
          y,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=2)

del train_set, y
gc.collect()

model.save_weights('dnn_keras.h5')

# Prediction
test_set = {
    APP: np.array(test_set.app),
    CHANNEL: np.array(test_set.channel),
    DEVICE: np.array(test_set.device),
    OS: np.array(test_set.os),
    CLICKS_BY_IP: np.array(test_set.clicks_by_ip),
    CLICKS_BY_IP_APP: np.array(test_set.clicks_by_ip_app),
    DAY_OF_WEEK: np.array(test_set.day_of_week),
    DAY_OF_YEAR: np.array(test_set.day_of_year)
}

sub[IS_ATTRIBUTED] = model.predict(
    test_set,
    batch_size=batch_size,
    verbose=2)

del test_set
gc.collect()

sub.to_csv('submission.csv', index=False)
