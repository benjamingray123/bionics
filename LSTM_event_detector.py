from pandas import read_csv
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def build_LSTM_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


df = read_csv('../data/labeled_testwalk.csv')

#  one-hot-encode GaitPhases and shape them accordingly. Shape=(10210, 2)
labels = df['GaitPhase'].values
label_encoder = LabelEncoder()
labels = to_categorical(label_encoder.fit_transform(labels).reshape(-1, 1),
                        dtype='float64')

t_labels = df['GaitPhase'].values[2000:3000]
t_labels = to_categorical(label_encoder.fit_transform(t_labels).reshape(-1, 1),
                          dtype='float64')

# reshape training and testing data. Shape=(10210, 1, 3)
data = df[['wx (rad/s)', 'wy (rad/s)',  'wz (rad/s)']].values.reshape(-1, 1, 3)

t_data = df[['wx (rad/s)', 'wy (rad/s)', 'wz (rad/s)']].values[2000:3000]
t_data = t_data.reshape(-1, 1, 3)

n_timesteps = data.shape[1]
n_features = data.shape[2]
n_outputs = labels.shape[1]

verbose, epochs, batch_size = 1, 15, 64

model = build_LSTM_model(n_timesteps, n_features, n_outputs)
model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
predictions = model.predict(t_data, batch_size=batch_size, verbose=1)
evaluations = model.evaluate(t_data, t_labels, batch_size=batch_size,
                             verbose=1)
