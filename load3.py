
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

load_data = np.loadtxt("./data.csv", delimiter=",")

label = load_data[:, 0]
data = load_data[:, 1:]
data = np.split(data, np.sort(np.unique(label, return_index=True)[1]))

data.pop(0)

new_data = []

for i, batch in enumerate(data) :
    if batch.shape[0] > 5 :
        splited_batch = np.split(np.flip(batch, axis=0), np.array([(i, i+6) for i in range(0, batch.shape[0] - 5)]).reshape(-1))
        splited_batch = list(map(lambda x : x.tolist(), splited_batch))

        for j, sample in enumerate(splited_batch) :
            splited_batch.pop(j)

        new_data += splited_batch

new_data = np.array(new_data)

print(new_data, new_data.shape)