# NumPy XOR Classifier From Scratch
A raw NumPy implementation of a multi-layer perceptron (MLP) neural network trained to solve the non-linear XOR logic gate problem. This project demonstrates backpropagation math and gradient descent without relying on high-level AI frameworks like PyTorch or TensorFlow.

## 🚀 Key Learning Objectives Achieved
- **Matrix Dimension Tracking**: Handled dot products (`np.dot`) and matrix transposes (`.T`) manually to align input features with hidden layer weights.
- **Activation Functions**: Implemented the Sigmoid function to squish outputs between `0` and `1`, adding non-linearity to the network.
- **Backpropagation Math**: Applied the mathematical chain rule using the derivative of the Sigmoid function to propagate errors backward.
- **Optimization**: Used Gradient Descent to iteratively adjust weights and biases over thousands of epochs to minimize Mean Squared Error (MSE).

## 🛠️ Network Architecture
- **Input Layer**: 2 nodes (handling binary combinations: `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`)
- **Hidden Layer**: 3 nodes with Sigmoid Activation (to map inputs into a higher-dimensional space where they become linearly separable)
- **Output Layer**: 1 node with Sigmoid Activation (predicting a confidence score close to `0` or `1`)

## 💻 How to Run This Project
1. Clone this repository to your local machine.
2. Ensure you have NumPy installed: `pip install numpy`
3. Run the script: `python xor_nn.py`
