import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping 
from keras import regularizers
import sklearn
import numpy as np
import pandas as pd
import collections

df = pd.read_csv('train.csv')
print (df['PAID_NEXT_MONTH'].value_counts().to_dict())
target = to_categorical(df['PAID_NEXT_MONTH'].values)
print((target == 0).sum())
print(len(target))

predictors = df.drop('PAID_NEXT_MONTH', axis=1).values
n_cols = predictors.shape[1]
# print(predictors)

model = Sequential()
model.add(Dense(25, activation='relu', input_shape = (n_cols,)))
model.add(Dense(50, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_weight = {0: 35, 1: 9}
model.fit(predictors, target, epochs = 10, validation_split=0.3, class_weight=class_weight)

test = pd.read_csv('test.csv')
x_new = test.drop('PAID_NEXT_MONTH', axis=1).values
y_new = model.predict_classes(x_new)


for i in range(20):
    print("id %s \t Predicted=%s"  %(x_new[i][0], y_new[i]))
