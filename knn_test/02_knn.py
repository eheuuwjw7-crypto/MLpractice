from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._repr_html import estimator

x_train = [0,0,1],[1,1,0],[3,10,10],[4,11,12]
y_train = [0.1,0.2,0.3,0.4]
x_test = [[3,11,10]]

estimator = KNeighborsRegressor(n_neighbors = 3)
estimator.fit(x_train,y_train)

y_pre = estimator.predict(x_test)

print(f'预测值为{y_pre}')