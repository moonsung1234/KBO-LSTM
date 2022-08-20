
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle

load_data = np.loadtxt("./data.csv", delimiter=",")

label = load_data[:, 0]
data = load_data[:, 1:]
data = np.split(data, np.sort(np.unique(label, return_index=True)[1]))

data.pop(0)

new_data = []
data_len = 5

for i, batch in enumerate(data) :
    if batch.shape[0] > data_len :
        splited_batch = np.split(np.flip(batch, axis=0), np.array([(i, i+data_len + 1) for i in range(0, batch.shape[0] - data_len)]).reshape(-1))
        splited_batch = list(map(lambda x : x.tolist(), splited_batch))

        for j, sample in enumerate(splited_batch) :
            splited_batch.pop(j)

        new_data += splited_batch

new_data = np.array(new_data)
# new_data = np.concatenate((new_data[:, :, :1] * 1000, new_data[:, :, 1:]), axis=2)

# const_dic = {
#     "AVG" : 1000, # under the 1
#     "G" : 1,   
#     "PA" : 1,   
#     "AB" : 1,    
#     "R" : 1,   
#     "H" : 1,   
#     "TWB" : 1, 
#     "THB" : 1,
#     "HR" : 1,  
#     "TB" : 1, 
#     "RBI" : 1, 
#     "SB" : 1, 
#     "CS" : 1, 
#     "BB" : 1, 
#     "HBP" : 1, 
#     "SO" : 1, 
#     "GDP" : 1, 
#     "SLG" : 1, # under the 1
#     "OBP" : 1, # under the 1
#     "E" : 1
# }

# new_data[:, :, 0] *= const_dic["AVG"]
# new_data[:, :, 1] *= const_dic["G"]
# new_data[:, :, 2] *= const_dic["PA"]
# new_data[:, :, 3] *= const_dic["AB"]
# new_data[:, :, 4] *= const_dic["R"]
# new_data[:, :, 5] *= const_dic["H"]
# new_data[:, :, 6] *= const_dic["TWB"]
# new_data[:, :, 7] *= const_dic["THB"]
# new_data[:, :, 8] *= const_dic["HR"]
# new_data[:, :, 9] *= const_dic["TB"]
# new_data[:, :, 10] *= const_dic["RBI"]
# new_data[:, :, 11] *= const_dic["SB"]
# new_data[:, :, 12] *= const_dic["CS"]
# new_data[:, :, 13] *= const_dic["BB"]
# new_data[:, :, 14] *= const_dic["HBP"]
# new_data[:, :, 15] *= const_dic["SO"]
# new_data[:, :, 16] *= const_dic["GDP"]
# new_data[:, :, 17] *= const_dic["SLG"]
# new_data[:, :, 18] *= const_dic["OBP"]
# new_data[:, :, 19] *= const_dic["E"]

# new_data[:, :, 0] *= 10

scaler_list = []

for i in range(new_data.shape[-1]) :
    scaler = StandardScaler()

    new_data_shape = new_data[:, :, i].shape
    new_data[:, :, i] = scaler.fit_transform(new_data[:, :, i].flatten().reshape(-1, 1)).reshape(new_data_shape)

    scaler_list.append(scaler)

train_x = new_data[:, :-1]
train_t = new_data[:, -1, 0]

train_input, test_input, train_target, test_target = train_test_split(train_x, train_t, test_size=0.2, random_state=2022)

print(train_input.shape, train_target.shape)
print(train_input[0], train_target[0])

model = keras.Sequential([
    keras.layers.LSTM(50, input_shape=(data_len, 20), return_sequences=True),
    keras.layers.LSTM(64),
    # keras.layers.LSTM(200),
    # keras.layers.Dense(1000, input_shape=(data_len, 1), activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    # keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse",
)

check_point = keras.callbacks.ModelCheckpoint("best_lstm.h5")
# early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.summary()

history = model.fit(
    train_input,
    train_target,
    epochs=100,
    batch_size=10,
    validation_data=(test_input, test_target),
    callbacks=[check_point]
)

predict = model.predict(train_input[:10])

# print(predict, train_target[:10])

scaler = scaler_list[0]

print(scaler.inverse_transform(predict), "\n", scaler.inverse_transform(train_target[:10].reshape(-1, 1)))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.xlabel("loss")
plt.legend(["train", "val"])
plt.show()