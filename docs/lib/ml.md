### Loss Functions:
- `binary_crossentropy(y_true, y_pred)`
- `binary_focal_loss(y_true, y_pred, gamma = 2.0, alpha = 0.25)`
- `categorical_crossentropy(y_true, y_pred)`
- `cosine_similarity(y_true, y_pred)`
- `dice_loss(y_true, y_pred)`
- `focal_loss(y_true, y_pred, gamma = 2.0, alpha = 0.25)`
- `hinge_loss(y_true, y_pred)`
- `huber_loss(y_true, y_pred, delta = 1.0)`
- `kldivergence(y_true, y_pred)` (Kullback-Leibler divergence)
- `log_cosh(y_true, y_pred)`
- `mae(y_true, y_pred)` (Mean Absolute Error)
- `mse(y_true, y_pred)` (Mean Squared Error)
- `quantile_loss(y_true, y_pred, quantile = 0.5)`

### Activation Functions:
- `elu(x, alpha = 1.0)` (Exponential Linear Unit)
- `gelu(x)` (Gaussian Error Linear Unit)
- `gelu_approx(x)`
- `relu(x)` (Rectified Linear Unit)
- `prelu(x, alpha = 0.01)` (Parametric ReLU)
- `sigmoid(x)`
- `softmax(inputs)`
- `softplus(x)`
- `softsign(x)`
- `selu(x)` (Scaled Exponential Linear Unit)
- `swish(x, beta = 1.0)`
- `leaky_relu(x, alpha = 0.01)`
- `linear(x)`
- `tanh(x)`
- `tanh_shrink(x)`

### Optimizers:
- `rmsprop(weights, gradients, v, learning_rate = 0.001, decay_rate = 0.9)`
- `adadelta(weights, gradients, accum_grad, accum_update, rho = 0.95)`
- `adagrad(weights, gradients, v, learning_rate = 0.01)`
- `adam(weights, gradients, m, v, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, t = 1)`
- `adamax(weights, gradients, m, v, learning_rate = 0.002, beta1 = 0.9, beta2 = 0.999, t = 1)`
- `nadam(weights, gradients, m, v, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, t = 1)`
- `sgd(weights, gradients, velocity, learning_rate = 0.01, momentum = 0.0)`
- `nesterov_sgd(weights, gradients, velocity, learning_rate = 0.01, momentum = 0.9)`

### Regularization:
- `dropout(x, dropout_rate = 0.5)`
- `elastic_net(weights, lambda1 = 0.01, lambda2 = 0.01)`
- `l1_regularization(weights, lambda = 0.01)`
- `l2_regularization(weights, lambda = 0.01)`
- `weight_decay(weights, lambda = 0.01)`

