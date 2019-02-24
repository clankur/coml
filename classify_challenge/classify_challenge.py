import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers

import sklearn
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import collections

def load_data(training_file):
    df = pd.read_csv(training_file)
    target = to_categorical(df['PAID_NEXT_MONTH'].values)
    predictors = df.drop('PAID_NEXT_MONTH', axis=1).values
    predictors = normalize(predictors, norm='l2')
    return (predictors, target)

def create_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def write_submission(filename):
    f = open(filename, 'w')
    f.write("ID,PAID_NEXT_MONTH\n")
    for i in range(len(x_new)):
        f.write("%s, %s \n" % (int(x_new[i][0]), y_new[i]))
    f.close()

predictors, target = load_data('train.csv')
n_cols = predictors.shape[1]
model = create_model()

class_weight = {0: 7, 1: 3}
model.fit(predictors, target, epochs=10,
          validation_split=0.3, class_weight=class_weight)

test = pd.read_csv('test.csv')
x_new = test.drop('PAID_NEXT_MONTH', axis=1).values
y_new = model.predict_classes(x_new)

write_submission('submission.csv')
df = pd.read_csv('submission.csv')
print(df['PAID_NEXT_MONTH'].value_counts().to_dict())

