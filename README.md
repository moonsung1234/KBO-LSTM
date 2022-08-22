
# KBO-LSTM

<br/>

    선수들의 연간 지표를 활용하여 다음 년도의 성적을 예상하는 프로그램을 만들었다. 

<br/>

## Feature (example)

|AVG|G|PA|AB|R|H|2B|3B|HR|TB|RBI|SB|CS|BB|HBP|SO|GDP|SLG|OBP|E|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.273|99|353|319|33|87|19|3|5|127|32|5|2|26|5|46|12|0.398|0.335|3|

- 20개의 feature 를 가진 데이터를 학습에 이용하였다. 

<br/>

-----

<br/>

## Predict

<img 
    src="https://user-images.githubusercontent.com/71556009/185950621-14c1b60d-4509-4efb-aa71-0e158639567a.jpg"
    width="500px" 
    height="500px"
/>
- 본 예시로 사용한 선수 : 김현수 (LG)

![model_graph](https://user-images.githubusercontent.com/71556009/185950647-517409ee-da4b-4e76-8f6e-098af75bdd0e.png)

> 1. 마지막 성적을 예측한 그래프 (red)
> 2. 중간중간 성적을 예측한 그래프 (red)

<br/>

-----

<br/>

## Problem

- predict 값은 타율만 내보내는데, 타율은 보편적으로 엄청난 차이를 보이지 않고 0.1 ~ 0.3 사이의 값에 분포하기 때문에 바른 학습이 힘들다.
- super overfitting 이 일어난다. (미해결)

<br/>

-----

<br/>

## More

- feature 공학을 통해 feature 수를 늘리면 성능이 향상되는지 테스트 해볼 여부가 있다.
- StandardScaler 를 통해 학습 데이터를 전처리하였는데, 그 밖의 전처리 방법을 통해 성능 향상을 꾀어볼 수 있다. (실제로 스케일러에 따라 성능차이가 극단적으로 나는 경우가 많았음.)

<br/>