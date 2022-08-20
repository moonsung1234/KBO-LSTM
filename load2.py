
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import numpy as np

load_data = np.loadtxt("./data.csv", delimiter=",")

label = load_data[:, 0]
data = load_data[:, 1:]
data = np.split(data, np.sort(np.unique(label, return_index=True)[1]))

data.pop(0)

new_data = []

for i, batch in enumerate(data) :
    if batch.shape[0] > 3 :
        splited_batch = np.split(np.flip(batch, axis=0), np.array([(i, i+4) for i in range(0, batch.shape[0] - 3)]).reshape(-1))
        splited_batch = list(map(lambda x : x.tolist(), splited_batch))

        for j, sample in enumerate(splited_batch) :
            splited_batch.pop(j)

        new_data += splited_batch

new_data = np.array(new_data)
new_data = np.concatenate((new_data[:, :, :1], new_data[:, :, 1:]), axis=2)

# 처음 데이터 학습시키고 predict 부분에서 inverse_transform 시킬때 처음 데이터에 합쳐서 변환 ㄱㄱ
scaler = MinMaxScaler()
scaler.fit(new_data[0])

for i, sample in enumerate(new_data) :
    new_data[i] = scaler.transform(sample)

train_x = new_data[:, :-1]
train_t = new_data[:, -1, 0]
# train_t = train_t.reshape(train_t.shape[0], train_t.shape[1], 1)

train_input, test_input, train_target, test_target = train_test_split(train_x, train_t, test_size=0.2, random_state=2022)

print(train_input.shape, train_target.shape)
print(train_input[0], train_target[0])

model = keras.Sequential([
    keras.layers.LSTM(3, input_shape=(3, 20)),
    # keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam", 
    loss="mse", 
    metrics=["accuracy"]
)

check_point = keras.callbacks.ModelCheckpoint("best_lstm.h5")
# early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.summary()

model.fit(
    train_input, 
    train_target, 
    epochs=100, 
    batch_size=100, 
    validation_data=(test_input, test_target),
    callbacks=[check_point]
)

predict = model.predict(train_input[0].reshape(1, 3, 20))

structure = new_data[0].copy()
structure[0, 0] = train_target[0]
structure[1, 0] = predict[0, 0]

final_predict = scaler.inverse_transform(structure)

print(final_predict[0, 0], final_predict[1, 0])

