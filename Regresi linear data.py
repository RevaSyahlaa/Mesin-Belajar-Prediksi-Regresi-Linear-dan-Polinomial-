import numpy as np
from sklearn.linear_model import LinearRegression

# data yang digunakan
# x = data
# y = target

x = [[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]
y = [1, 4, 7, 10, 13, 16, 19, 21, 24, 27]

regr = LinearRegression().fit(x,y)
regr.score(x,y)

predict = np.array([[6]])

print ("Prediksi")
print ("Input = ", predict)
print ("Output= ", regr.predict(predict))
