import numpy as np


# Layer.
class Layer:
    def __init__(self, layer_neurons, activation="linear"):

        self.layer_neurons = layer_neurons
        self.act = activation
        self.activation = getattr(self, "_" + activation)
        self.activation_derivative = getattr(self, "_" + activation + "_derivative")
        self.preActivation = None
        self.posActivation = None

    def _linear(self, x):
        return x

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return 2 * self._sigmoid(2 * x) - 1

    def _relu(self, x):
        return np.maximum(x, 0)

    def _hardTanh(self, x):
        return np.clip(x, -1, 1)

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _tanh_derivative(self, x):
        return 1 - x**2

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _hardTanh_derivative(self, x):
        return ((x > -1) & (x < 1)).astype(float)

    def _softmax_derivative(self, x):
        return np.ones_like(x)


# Neural Network.
class NeuralNetwork:

    def __init__(self, input_neurons):

        # Input layer.
        input_layer = Layer(input_neurons)

        # Layers.
        self.layers = list()
        self.layers.append(input_layer)

        # Weights & biases.
        self.weights = list()
        self.biases = list()

        # Loss function.
        self.loss = None

        self.learning_rate = 1

        # Training history.
        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def add_layer(self, layer_neurons, activation):
        layer = Layer(layer_neurons, activation=activation)
        self.layers.append(layer)

    def predict(self, vector_x, print_forward=False):

        # Input layer.
        self.layers[0].preActivation = vector_x
        self.layers[0].posActivation = vector_x

        # Hidden + Output Layers.
        for i in range(1, len(self.layers)):

            # a = w1*x1 + w2*x2 + ... + b
            self.layers[i].preActivation = (
                np.dot(vector_x, self.weights[i - 1]) + self.biases[i - 1]
            )

            # h(a) = h(w1*x1 + w2*x2 + ... + b) -> h is the activation function.
            self.layers[i].posActivation = self.layers[i].activation(
                self.layers[i].preActivation
            )

            # Update vector x.
            vector_x = self.layers[i].posActivation

            # Print values.
            if print_forward:
                print("")
                print("Hidden Layer: ", i)
                print(
                    "Activation: ", self.layers[i].activation.__name__.replace("_", "")
                )
                print("Pre Activation: ", self.layers[i].preActivation)
                print("Pos Activation: ", self.layers[i].posActivation)

        return vector_x

    def initialize_nn(self, loss="cross_entropy", learning_rate=1):
        self.loss = getattr(self, "_" + loss)
        self.loss_derivative = getattr(self, "_" + loss + "_derivative")
        self.learning_rate = learning_rate

        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i].layer_neurons
            if self.layers[i + 1].act == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            matrix_weights = (
                np.random.randn(
                    self.layers[i].layer_neurons, self.layers[i + 1].layer_neurons
                )
                * scale
            )
            self.weights.append(matrix_weights)
            self.biases.append(np.zeros(self.layers[i + 1].layer_neurons))

        num_params = sum(w.size for w in self.weights) + sum(
            b.size for b in self.biases
        )
        print("Number of parameters:", num_params)

    def fit(self, X, Y, epochs=10, batch_size=32, validation_data=None):
        n_samples = len(X)
        n_classes = self.layers[-1].layer_neurons

        for epoch in range(epochs):
            # Shuffle training data each epoch.
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            epoch_loss = 0.0
            correct = 0

            # Mini-batch gradient descent.
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                actual_batch = end - start

                # Accumulate weight / bias gradients across the batch.
                weight_grads = [np.zeros_like(w) for w in self.weights]
                bias_grads = [np.zeros_like(b) for b in self.biases]

                for x, y_int in zip(X_batch, Y_batch):
                    # One-hot encode the label.
                    y_true = np.zeros(n_classes)
                    y_true[y_int] = 1.0

                    y_pred = self.predict(x)

                    sample_loss = self.loss(y_true, y_pred)
                    epoch_loss += np.sum(sample_loss)

                    predicted_class = np.argmax(y_pred)
                    if predicted_class == y_int:
                        correct += 1

                    # Backprop returns per-layer gradients (output → input order).
                    gradients = self.back_propagation(y_true, y_pred)

                    for i in range(len(gradients)):
                        layer_idx = len(self.layers) - 1 - i
                        prev_pos = self.layers[layer_idx - 1].posActivation
                        weight_grads[-(i + 1)] += np.outer(prev_pos, gradients[i])
                        bias_grads[-(i + 1)] += gradients[i]

                clip = 5.0
                for i in range(len(self.weights)):
                    wg = np.clip(weight_grads[i] / actual_batch, -clip, clip)
                    bg = np.clip(bias_grads[i] / actual_batch, -clip, clip)
                    self.weights[i] -= self.learning_rate * wg
                    self.biases[i] -= self.learning_rate * bg

            train_loss = epoch_loss / n_samples
            train_acc = correct / n_samples
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)

            # Validation.
            if validation_data is not None:
                X_val, Y_val = validation_data
                val_loss, val_acc = self._evaluate(X_val, Y_val)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

    def _evaluate(self, X, Y):
        n_classes = self.layers[-1].layer_neurons
        total_loss = 0.0
        correct = 0
        for x, y_int in zip(X, Y):
            y_true = np.zeros(n_classes)
            y_true[y_int] = 1.0
            y_pred = self.predict(x)
            total_loss += np.sum(self.loss(y_true, y_pred))
            if np.argmax(y_pred) == y_int:
                correct += 1
        return total_loss / len(X), correct / len(X)

    def back_propagation(self, y_true, y_predicted):
        gradients = []

        # Output layer delta.
        # delta = dL/dy_pred * dy_pred/dz, dL is the loss derivative, dy_pred/dz is the activation derivative.
        last_layer = self.layers[-1]
        loss_grad = self.loss_derivative(y_true, y_predicted)
        delta = loss_grad * last_layer.activation_derivative(last_layer.posActivation)
        gradients.append(delta)

        # Hidden layer deltas.
        for i in range(2, len(self.layers)):
            # delta(i-1) = d(delta_i)/dz = d(delta_i)/dy_pred * dy_pred/dz
            layer = self.layers[-i]
            delta = np.dot(delta, self.weights[-i + 1].T)
            delta = delta * layer.activation_derivative(layer.preActivation)
            gradients.append(delta)

        return gradients

    def _MSE(self, y_true, y_predicted):
        return (y_true - y_predicted) ** 2

    def _MSE_derivative(self, y_true, y_predicted):
        return -2 * (y_true - y_predicted)

    def _cross_entropy(self, y_true, y_predicted):
        eps = 1e-12
        return -y_true * np.log(y_predicted + eps)

    def _cross_entropy_derivative(self, y_true, y_predicted):
        eps = 1e-12
        return -y_true / (y_predicted + eps)
