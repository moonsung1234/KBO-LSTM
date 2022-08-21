
from model import Model
import numpy as np

model = Model()
model.load()

predict_input = np.array([])
predict_value = model.predict(predict_input.reshape(1, predict_input.shape[0], predict_input.shape[1]))

print(predict_value)