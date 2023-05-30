import numpy as np
import math

learning_rate = 0.000001
batch_size = 50
num_epochs = 100000


class RegressionModel:

    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    def f(self, x1, x2):
        y = self.a * np.power(x2, 2) * x1
        y += self.b * np.power(x2, 2)
        y += self.c * x1
        y += self.d
        return y

    def grads_f(self, x1, x2, y):
        y_hat = self.f(x1, x2)
        d_y = 2 * (y_hat - y)
        da = np.power(x2, 2) * x1 * d_y
        db = np.power(x2, 2) * d_y
        dc = x1 * d_y
        da = np.sum(da, axis=0, keepdims=False) / batch_size
        db = np.sum(db, axis=0, keepdims=False) / batch_size
        dc = np.sum(dc, axis=0, keepdims=False) / batch_size
        dd = np.sum(d_y, axis=0, keepdims=False) / batch_size
        grads = {'da': da, 'db': db, 'dc': dc, 'dd': dd}
        return grads

    def loss(self, x1, x2, y):
        y_hat = self.f(x1, x2)
        loss = np.power(y - y_hat, 2)
        loss = np.sum(loss, axis=0, keepdims=False) / batch_size
        return loss

    def OptimizerStep(self, x1, x2, y):
        grads = self.grads_f(x1, x2, y)
        self.a -= learning_rate * grads['da']
        self.b -= learning_rate * grads['db']
        self.c -= learning_rate * grads['dc']
        self.d -= learning_rate * grads['dd']

    def SGD(self, x1, x2, y):
        number_of_batches = math.floor(len(y) / batch_size)
        for i in range(num_epochs):
            for j in range(number_of_batches):
                x1_batch = x1[j * batch_size: (j + 1) * batch_size]
                x2_batch = x2[j * batch_size: (j + 1) * batch_size]
                y_batch = y[j * batch_size: (j + 1) * batch_size]
                print("loss: ", self.loss(x1_batch, x2_batch, y_batch))
                self.OptimizerStep(x1_batch, x2_batch, y_batch)


if __name__ == "__main__":
    data = np.load("data.npz")
    x1 = np.array(data["x1"])
    x2 = np.array(data["x2"])
    y = np.array(data['y'])

    x1_cross_validation = x1[math.floor(len(x1) * 0.9):]
    x2_cross_validation = x2[math.floor(len(x2) * 0.9):]
    y_cross_validation = y[math.floor(len(y) * 0.9):]

    x1 = x1[:math.floor(len(x1) * 0.9)]
    x2 = x2[:math.floor(len(x2) * 0.9)]
    y = y[:math.floor(len(y) * 0.9)]

    model = RegressionModel()
    model.SGD(x1, x2, y)

    x1_test = np.array(data["x1_test"])
    x2_test = np.array(data["x2_test"])
    y_test = np.array(data["y_test"])

    # print(model.loss(x1_cross_validation, x2_cross_validation, y_cross_validation))
    print(model.loss(x1_test, x2_test, y_test))
