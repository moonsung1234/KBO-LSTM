
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Model :
    def __init__(self, data_file_path="./data.csv", data_length=5, scaler_file_path="scaler.pickle", model_file_path="./best_lstm.h5") :
        self.data_file_path = data_file_path
        self.data_length = data_length

        self.scaler_file_path = scaler_file_path
        self.scaler_list = []

        self.model_file_path = model_file_path
        self.history = None

    def __replace(self) :
        # load data
        load_data = np.loadtxt(self.data_file_path, delimiter=",")

        # group data by each same players
        label = load_data[:, 0]
        data = load_data[:, 1:]
        data = np.split(data, np.sort(np.unique(label, return_index=True)[1]))

        data.pop(0)

        # make time-series data
        new_data = []

        for i, batch in enumerate(data) :
            if batch.shape[0] > self.data_length :
                splited_batch = np.split(np.flip(batch, axis=0), np.array([(i, i+self.data_length + 1) for i in range(0, batch.shape[0] - self.data_length)]).reshape(-1))
                splited_batch = list(map(lambda x : x.tolist(), splited_batch))

                for j, _ in enumerate(splited_batch) :
                    splited_batch.pop(j)

                new_data += splited_batch

        new_data = np.array(new_data)

        # # make features
        # poly = PolynomialFeatures()
        
        # for i in range(new_data.shape[-1]) :    
        #     new_data_shape = new_data.shape
        #     new_data = poly.fit_transform(new_data.reshape())

        # scale data
        for i in range(new_data.shape[-1]) :
            scaler = StandardScaler()

            new_data_shape = new_data[:, :, i].shape
            new_data[:, :, i] = scaler.fit_transform(new_data[:, :, i].flatten().reshape(-1, 1)).reshape(new_data_shape)

            self.scaler_list.append(scaler)

        # save scaler list
        with open(self.scaler_file_path, "wb") as fp :
            pickle.dump(self.scaler_list, fp)

        return new_data

    def get_data(self) :
        return self.__replace()

    def load(self) :
        self.model = keras.models.load_model(self.model_file_path)
        
        with open(self.scaler_file_path, "rb") as fp :
            self.scaler_list = pickle.load(fp)

    def train(self) :
        data = self.__replace()

        train_x = data[:, :-1]
        train_t = data[:, -1, 0]

        # split train data
        train_input, test_input, train_target, test_target = train_test_split(train_x, train_t, test_size=0.2, random_state=2022)

        # make model (default)
        self.model = keras.Sequential([
            keras.layers.LSTM(50, input_shape=(train_input.shape[1], train_input.shape[2]), return_sequences=True),
            keras.layers.LSTM(64),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(1, activation="linear")
        ])

        self.model.compile(
            optimizer="adam",
            loss="mse",
        )

        check_point = keras.callbacks.ModelCheckpoint(self.model_file_path)
        # early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

        self.model.summary()

        # fit model
        self.history = self.model.fit(
            train_input,
            train_target,
            epochs=100,
            batch_size=10,
            validation_data=(test_input, test_target),
            callbacks=[check_point]
        )

    def show(self) :
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.xlabel("epoch")
        plt.xlabel("loss")
        plt.legend(["train", "val"])
        plt.show()

    def predict(self, predict_input) :
        new_data = predict_input.copy()

        # scale data
        for i in range(new_data.shape[-1]) :
            scaler = self.scaler_list[i]

            new_data_shape = new_data[:, :, i].shape
            new_data[:, :, i] = scaler.fit_transform(new_data[:, :, i].flatten().reshape(-1, 1)).reshape(new_data_shape)

        predict = self.model.predict(new_data)
        scaler = self.scaler_list[0]

        return scaler.inverse_transform(predict)
