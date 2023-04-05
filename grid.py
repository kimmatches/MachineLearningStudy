import pandas as pd

#Tree Depth Node수 Gini
wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
# 데이터양이 충분할 때 두개로 쪼갬
# 검증세트
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
# 테스트 세트
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)

import numpy as np

print(np.mean(scores['test_score']))

#하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
#한번에 돌리기
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

#랜덤 서치