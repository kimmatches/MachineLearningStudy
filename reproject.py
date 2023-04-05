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
# fp = 연료 단가 ep = 통합 가격
ep = pd.read_excel('ep.xlsx', engine="openpyxl", usecols=[3],  header = 1)
fp = pd.read_excel('fp.xlsx', engine="openpyxl", usecols="B,E,F",  header = 1)

print(ep, fp)

input = fp[['원자력', '유류', 'LNG']].to_numpy()
target = ep.to_numpy()

print(input)
print(target)



plt.plot(input[:, 0])
plt.plot(input[:, 1])
plt.plot(input[:, 2])
plt.plot(target * 10000)
plt.show()


input = input[:,2:3]

poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(input)
train_poly = poly.transform(input)
print(np.shape(train_poly))
# degree=3 -> 0.8903493033849034
#             0.928216648665078
# degree=4 -> 0.9004946815075843
#             0.9121429396303544
# degree=5 -> 0.9147073737409539
#             0.8799654898265795

tr_in, ts_in, tr_out, ts_out = train_test_split(
    train_poly, target, test_size=0.5)

lr=LinearRegression()
lr.fit(tr_in, tr_out)
print(lr.score(tr_in, tr_out))
print(lr.score(ts_in, ts_out))