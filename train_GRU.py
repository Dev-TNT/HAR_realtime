import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

# read data
data_standing = pd.read_csv('standing.csv')
data_sitting = pd.read_csv('sitting.csv')
data_waving = pd.read_csv('waving.csv')
data_swat = pd.read_csv('swat.csv')
data_drinking = pd.read_csv('drinking.csv')
data_clapping = pd.read_csv('clapping.csv')
data_Tpose = pd.read_csv('Tpose.csv')

'''
data_crossing = pd.read_csv('crossing.csv')
data_pushup = pd.read_csv('pushup.csv')
'''

# data management
actions = {
    "standing": data_standing,
    "sitting": data_sitting,
    "waving": data_waving,
    "clapping": data_clapping,
#    "crossing": data_crossing,
    "drinking": data_drinking,
#    "pushup": data_pushup,
    "swat": data_swat,
    "Tpose": data_Tpose
}

# label recognition
label_map = {
    action: i for i, action in enumerate(actions.keys())
}

# CREATE SEQUENCE DATA
X = []
Y = []

sequence_length = 30
stride = 5

for action, data in actions.items():
    dataset = data.values
    n_sample = len(dataset)

    # CREATE SEQUENCE
    for i in range(sequence_length, n_sample, stride):

        sequence = dataset[i-sequence_length:i]

        X.append(sequence)

        Y.append(label_map[action])

# NUMPY ARRAY
X = np.array(X)
Y = np.array(Y)

# TRAIN TEST SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# BUILD MODEL
model = Sequential()

# LSTM
'''
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# GRU model
'''
# Sử dụng Bidirectional GRU để học được cả ngữ cảnh trước và sau
model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(BatchNormalization()) # Giúp ổn định training
model.add(Dropout(0.3))

model.add(Bidirectional(GRU(64)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# train
model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)

#save
model.save("GRU_action_model_1.keras")

print("MODEL SAVED")