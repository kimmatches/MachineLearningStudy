from sklearn.linear_model import LinearRegression
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


import warnings
warnings.simplefilter("ignore")
# fp = input data, ep = target data (정답)

ep = pd.read_excel('ep.xlsx', engine="openpyxl", usecols=[3],  header = 1)
fp = pd.read_excel('fp.xlsx', engine="openpyxl", usecols="B,E,F",  header = 1)
# 엑셀에서 필요한 데이터만 가져오기 위해 데이터 분석 엑셀 파트 지정
# fp = 연료 단가 ep = 통합 가격
print(ep, fp)

# 필요한 데이터들을 배열로 만들어줌 이름
input = fp[['원자력', '유류', 'LNG']].to_numpy()
target = ep.to_numpy()

# print(input)
# print(target)


# 현재 데이터를 그래프로 보기 위해 파트 지정 후 출력
plt.plot(input[:, 0])
plt.plot(input[:, 1])
plt.plot(input[:, 2])
plt.plot(target * 10000)
plt.show()


poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(input)
train_poly = poly.transform(input)
print(np.shape(train_poly))
# 여러 차수로 실험
# 차수가 3일 때는 훈련 정확도와 테스트 정확도에서 오차 범위가 0.1~0.2 내외지만
# 4,5에서는 오차범위가 매우 크거나 과적합의 결과를 얻을 수 있음
# degree=3 ->0.9457157568230259
#            0.9185429379929216
# degree=4 -> 0.9694422299100093
#             0.5249031252718142
# degree=5 -> 0.8177909728810917
#            -349.4463933007212
#test_size는 0.5에 맞춰야 제일 정확한 정확도를 얻을 수 있다.
tr_in, ts_in, tr_out, ts_out = train_test_split(
    train_poly, target, test_size=0.5)

lr=LinearRegression()
lr.fit(tr_in, tr_out)
print(lr.score(tr_in, tr_out))
print(lr.score(ts_in, ts_out))