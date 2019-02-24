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

def load_data(training_file, target_name):
    df = pd.read_csv(training_file)
    target = to_categorical(df[target_name].values)
    predictors = df.drop(target_name, axis=1).values
    predictors = normalize(predictors, norm='l2')
    return (predictors, target)

def create_model(n_cols):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(200, activation='relu', input_shape=(n_cols,)))
   
    model.add(Dense(2, activation='softmax'))
    optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def write_submission(filename, id, target_name, x_new, y_new):
    f = open(filename, 'w')
    f.write("%s,%s\n" % (id, target_name))
    for i in range(len(x_new)):
        f.write("%s, %s \n" % (int(x_new[i][0]), y_new[i]))
    f.close()

infile = 'train.csv'
testfile = 'test.csv'
outfile = 'submission.csv'
target_name = 'PAID_NEXT_MONTH'

predictors, target = load_data(infile, target_name)
n_cols = predictors.shape[1]
model = create_model(n_cols)

class_weight = {0: 56, 1: 24}
model.fit(predictors, target, epochs=50,
          validation_split=0.3, class_weight=class_weight)

test = pd.read_csv(testfile)
x_new = test.drop(target_name, axis=1).values
y_new = model.predict_classes(x_new)

write_submission(outfile, 'ID', target_name, x_new, y_new)


df = pd.read_csv(outfile)
print(df[target_name].value_counts().to_dict())

