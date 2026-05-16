from sklearn.preprocessing import MinMaxScaler

x_train = [[90, 2, 10, 40],[60, 4, 15, 45],[30, 6, 20, 47],[80, 8, 25, 49],[50, 10, 30, 53],[10, 12, 35, 57]]

transfer = MinMaxScaler(feature_range=(0, 1))
x_train = transfer.fit_transform(x_train)

print(x_train)