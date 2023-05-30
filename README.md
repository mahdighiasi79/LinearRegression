# LinearRegression
In this project, we aimed to learn a specific function's coefficients.
We have a data matrix in the "data.npz" file which contains three attributes: x1, x2, y.

These features follow the relation written below:
y = (4 * x_2^2 * x_1) + (2 * x_2^2) + (3 * x_1) + 1

In our regression task, we assume the function as bellow:
y = (a * x_2^2 * x_1) + (b * x_2^2) + (c * x_1) + d

Then, we will try to learn the coefficients, a, b, c, and d by linear regression.
We initialized all the parameters equal to zero.
The Sum of Squared Errors (SSE) loss function has been used for this task.
And the Stochastic Gradient Descent (SGD) algorithm has been used for optimizing the cost function.

Then, we split the train data into two parts. 
The first part which contains 90% of the data is used for learning parameters.
And the second part which contains 10% of the data is used as the cross validation set for tuning the hyperparameters.
The test set is totally separated from our training data.

After training the regression model and testing it on the cross validation set, we ended up with bathc_size = 50, and learning_rate = 0.000001 hyperparameters.
As the data have been generated exactly by the mentioned function and the concept of overfitting doesn't exist, if you have more epochs, you have more accuracy.
I have run the training with 100,000 epochs and it gave a 0.0034064074434916546 loss value on the test set.
But running the regression with 1000 epochs will give you a reasonable accuracy.
