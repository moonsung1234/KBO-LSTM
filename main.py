
from model import Model
import matplotlib.pyplot as plt
import numpy as np

# model load
model = Model(
    data_file_path="./data/data.csv",
    scaler_file_path="./model/scaler.pickle",
    model_file_path="./model/best_lstm.h5"
)
model.load()
# model.train()
# model.show()

# LG 김현수 (예시)
predict_input = np.array([
    [
        [
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            0.273,
            99,
            353,
            319,
            33,
            87,
            19,
            3,
            5,
            127,
            32,
            5,
            2,
            26,
            5,
            46,
            12,
            0.398,
            0.335,
            3
        ],
        [
            0.357,
            126,
            558,
            470,
            83,
            168,
            34,
            5,
            9,
            239,
            89,
            13,
            8,
            80,
            5,
            40,
            12,
            0.509,
            0.454,
            1
        ],
        [
            0.357,
            133,
            572,
            482,
            97,
            172,
            31,
            6,
            23,
            284,
            104,
            6,
            6,
            80,
            4,
            59,
            7,
            0.589,
            0.448,
            3
        ],
        [
            0.317,
            132,
            565,
            473,
            88,
            150,
            29,
            0,
            24,
            251,
            89,
            4,
            8,
            78,
            6,
            64,
            9,
            0.531,
            0.414,
            3
        ],
        [
            0.301,
            130,
            561,
            475,
            71,
            143,
            25,
            2,
            13,
            211,
            91,
            5,
            3,
            71,
            6,
            63,
            15,
            0.444,
            0.392,
            7
        ],
        [
            0.291,
            122,
            491,
            437,
            47,
            127,
            17,
            1,
            7,
            167,
            65,
            6,
            3,
            46,
            3,
            50,
            9,
            0.382,
            0.358,
            4
        ],
        [
            0.302,
            122,
            510,
            434,
            63,
            131,
            23,
            1,
            16,
            204,
            90,
            2,
            4,
            62,
            2,
            71,
            6,
            0.47,
            0.382,
            2
        ],
        [
            0.322,
            125,
            528,
            463,
            75,
            149,
            26,
            0,
            17,
            226,
            90,
            2,
            0,
            53,
            7,
            45,
            10,
            0.488,
            0.396,
            1
        ],
        [
            0.326,
            141,
            630,
            512,
            103,
            167,
            26,
            0,
            28,
            277,
            121,
            11,
            5,
            101,
            8,
            63,
            13,
            0.541,
            0.438,
            1
        ],
        [
            0.362,
            117,
            511,
            453,
            95,
            164,
            39,
            2,
            20,
            267,
            101,
            1,
            3,
            47,
            1,
            61,
            9,
            0.589,
            0.415,
            5
        ],
        [
            0.304,
            140,
            595,
            526,
            75,
            160,
            37,
            0,
            11,
            230,
            82,
            3,
            1,
            54,
            6,
            52,
            11,
            0.437,
            0.37,
            1
        ],
        [
            0.331,
            142,
            619,
            547,
            98,
            181,
            35,
            2,
            22,
            286,
            119,
            0,
            2,
            63,
            2,
            53,
            9,
            0.523,
            0.397,
            1
        ],
        [
            0.285,
            140,
            595,
            506,
            70,
            144,
            23,
            1,
            17,
            220,
            96,
            3,
            0,
            77,
            3,
            42,
            7,
            0.435,
            0.376,
            0
        ],
        [
            0.282,
            103,
            448,
            390,
            67,
            110,
            20,
            2,
            22,
            200,
            85,
            2,
            1,
            52,
            5,
            43,
            4,
            0.513,
            0.374,
            2
        ]
    ]
])

# 1. 마지막 성적을 예측한 그래프 만들기
predict_value = model.predict(predict_input[:, -5:])
year = np.array([2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)

plt.subplot(2, 1, 1)
plt.plot(
    np.array([[2022], [2023]]), 
    np.concatenate((predict_input[:, -1, :1], predict_value), axis=0),
    color="red",
    marker="o"
)
plt.plot(
    year, 
    predict_input[0, -5:, :1], 
    color="blue", 
    marker="o"
)

for i, sample in enumerate(np.concatenate((predict_input[:, -1, :1], predict_value), axis=0).flatten().tolist()) :
    plt.text(
        np.array([[2022], [2023]])[i], 
        sample + 0.01,
        "%.2f" % sample,
        horizontalalignment="center",
        # verticalalignment="bottom",
        color="red"
    )

for i, sample in enumerate(predict_input[0, -5:, :1].flatten().tolist()) :
    plt.text(
        year[i], 
        sample + 0.01,
        "%.2f" % sample,
        horizontalalignment="center",
        # verticalalignment="top",
        color="blue"
    )

plt.xlabel("year")
plt.ylabel("batting average")

# 2. 중간중간 성적을 예측한 그래프 만들기
data = []
year = np.array([2011, 2012, 2013, 2014, 2015, 2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)

for i in range(predict_input.shape[1] - 4) :
    predict_value = model.predict(predict_input[:, i:i+5])

    data.append(predict_value[0, 0])

data = np.array(data).reshape(-1, 1)

plt.subplot(2, 1, 2)
plt.plot(
    year,
    data,
    color="red",
    marker="o"
)

for i, sample in enumerate(data.flatten().tolist()) :
    plt.text(
        year[i], 
        sample + 0.01,
        "%.2f" % sample,
        horizontalalignment="center",
        # verticalalignment="bottom",
        color="red"
    )

plt.xlabel("year")
plt.ylabel("batting average")

plt.show()