import numpy as np
import matplotlib.pyplot as plt

#머신러닝 비지도 학습 (혼자하는 딥러닝 머신러닝 책 참고)

fruits = np.load('fruits_300.npy')
# print(fruits.shape)

#reshape 모양을 바꾼다
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
#가로 100 세로 100 (2차원 배열) -> 1차원 배열 10000
# print(apple.shape)

#바나나는 구별할 수 있지만 나머지는 구별하기 어려운 것을 확인 할 수 있음
# plt.hist(np.mean(apple, axis=1), alpha=0.8)
# plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
# plt.hist(np.mean(banana, axis=1), alpha=0.8)
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()

# 픽셀 평균 히스토그램
# 각 픽셀 위치별로 평균을 냄 (과일 하나당 10000개의 픽셀) -> 밝기 비교
# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()

# 평균 이미지 그리기
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# axs[0].imshow(apple_mean, cmap='gray_r')
# axs[1].imshow(pineapple_mean, cmap='gray_r')
# axs[2].imshow(banana_mean, cmap='gray_r')
# plt.show()

# pineapple_mean을 다른 과일로 바꾸면 다른 과일도 볼 수 있다.
abs_diff = np.abs(fruits - pineapple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()