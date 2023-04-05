import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

fish1_data_size = 500
fish2_data_size = 400
fish3_data_size = 500
fish4_data_size = 400



a = np.load('data_target.npy')
#print(a)

#arr = np.split(a, [100])

#print(arr[0])
#print(arr[1])
#print(a.T)

#np.array_split()
b= np.split(a.T,2,axis=0)
fish_length = b[0]
fish_weight= b[1]
#fish_data = [[l, w] for l, w in zip(b[0],b[1])]
fish_data = np.load('data_input.npy')
fish_target = np.load('data_target.npy')



kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)


#t = kn.predict([[10, 200]])
#plt.scatter(10, 200, color='y')

plt.scatter(fish_data[:500, 0], fish_data[:500, 1])
plt.scatter(fish_data[500:900, 0], fish_data[500:900, 1])
plt.scatter(fish_data[900:1300, 0], fish_data[900:1300, 1])
plt.scatter(fish_data[1300:, 0], fish_data[1300:, 1])

fish1_data = fish_data[:500, 0], fish_data[:500, 1]
fish2_data = fish_data[500:900, 0], fish_data[500:900, 1]
fish3_data = fish_data[900:1300, 0], fish_data[900:1300, 1]
fish4_data = fish_data[1300:, 0], fish_data[1300:, 1]

fish_target2 = [0]*fish_data[:500, 0], fish_data[:500, 1] + [1]*fish_data[500:900, 0], fish_data[500:900, 1] + [3]*fish_data[900:1300, 0], fish_data[900:1300, 1] + [4]*fish_data[1300:, 0], fish_data[1300:, 1]

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target2, stratify=fish_target2, random_state=1800)

#train_input = fish_data[:1500]
#train_target = fish_target[:1500]

#test_input = fish_data[1500:]
#test_target = fish_target[1500:]

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input,test_target)
plt.show()
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std

distances, indexes = kn.kneighbors([0,0])
plt.scatter(train_scaled[:,0],train_input[:,1])
plt.scatter(0, 0, marker='^')
plt.scatter(train_scaled[indexes,0],
train_input[indexes, 1],marker='D')


#plt.scatter(fish_length, fish_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

