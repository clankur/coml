import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers

import sklearn
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(training_file, target_name):
    df = pd.read_csv(training_file)
    target = to_categorical(df[target_name].values)
    predictors = df.drop(target_name, axis=1).values
    predictors = normalize(predictors, norm='l2')
    return (predictors, target)

def create_model(n_cols, lr=0.001):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(200, activation='relu'))
   
    model.add(Dense(2, activation='softmax'))
    optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def plot_acc_loss(history, lr, disp_acc=False, disp_loss=False):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_set', 'val_set'], loc='upper left')
    plt.savefig('graphs/acc.png')
    if (disp_acc):
       plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss for LR=' +str(lr))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_set', 'val_set'], loc='upper left')
    plt.savefig('graphs/loss%s.png' % (lr))
    if (disp_loss):
       plt.show()

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
lr = 0.005

predictors, target = load_data(infile, target_name)
model = create_model(predictors.shape[1], lr)
class_weight = {0: 56, 1: 24}
history = model.fit(predictors, target, epochs=55, 
        validation_split=0.33, class_weight=class_weight, verbose=0)

plot_acc_loss(history, lr, True, True)

test = pd.read_csv(testfile)
x_new = test.drop(target_name, axis=1).values
y_new = model.predict_classes(x_new)
write_submission(outfile, 'ID', target_name, x_new, y_new)

df = pd.read_csv(outfile)
print(df[target_name].value_counts().to_dict())
