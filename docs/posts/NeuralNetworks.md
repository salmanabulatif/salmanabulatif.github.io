---
date:
  created: 2025-07-13
readtime: 5
authors:
    - salmanabulatif
---

# Neural Network

Neural networks rely on two key mathematical processes: **forward propagation** (for predictions) and **backward propagation** (for learning via gradient descent).

<!-- more -->

# The Math Behind Neural Network Forward and Backward Propagation

Neural networks rely on two key mathematical processes: **forward propagation** (for predictions) and **backward propagation** (for learning via gradient descent). In this post, we break down the core math behind these steps using a simple 3-layer neural network example. Hand-written calculations and diagrams from my notes are included for clarity.

---

## Network Architecture

We consider a neural network with:

- **Input layer:** 3 neurons (`x₁`, `x₂`, `x₃`)
- **Hidden layer:** 2 neurons (`h₄`, `h₅`)
- **Output layer:** 1 neuron (`o₆`)

```
Input Layer (x₁, x₂, x₃) → Hidden Layer (h₄, h₅) → Output Layer (o₆)
```

![image3](image3)

---

## 1. Forward Pass

The forward pass computes the output of each neuron layer-by-layer using weights (`ω`), biases (`b`), and activation functions (sigmoid here).

### Hidden Layer Calculations

For neuron 4 (`h₄`):

```math
a_4 = \sigma(\text{inputs} \cdot \text{weights} + b_4)
```

From the notes:

```math
a_4 = \frac{1}{1 + e^{-(-0.7)}} = 0.332
```

For neuron 5 (`h₅`):

```math
a_5 = \frac{1}{1 + e^{-0.1}} = 0.325
```

### Output Layer Calculation

```math
o_6 = \sigma(a_4 \omega_{46} + a_5 \omega_{56} + b_6)
```

```math
a_6 = \frac{1}{1 + e^{-(-0.206)}} = 0.194
```

![image2](image2)

---

## 2. Loss Calculation

The error (loss) is computed between the prediction (`o₆`) and the target value.

From the notes (target = 1):

```math
\text{Error} = 1 - 0.414 = 0.526
```

---

## 3. Backward Propagation

Backward propagation applies the chain rule to compute gradients for each weight and bias.

### Output Neuron Gradient (\(\delta_6\))

```math
\delta_6 = \text{Error} \times \sigma'(o_6)
```

where

```math
\sigma'(z) = \sigma(z)(1 - \sigma(z))
```

From the notes:

```math
\delta_6 = 0.414 \times (1 - 0.414) \times (1 - 0.414) = 0.131
```

### Hidden Neuron Gradients

For neuron 5 (\(\delta_5\)):

```math
\delta_5 = a_5(1 - a_5) \times (\delta_6 \cdot \omega_{56})
```

```math
\delta_5 = 0.325 \times (1 - 0.525) \times (0.2 \times 0.131) = 0.095
```

For neuron 4 (\(\delta_4\)):

```math
\delta_4 = a_4(1 - a_4) \times (\delta_6 \cdot \omega_{46})
```

```math
\delta_4 = 0.332 \times (1 - 0.332) \times (0.2 \times 0.131) = 0.095
```

![image1](image1)

---

## 4. Weight Updates

Weights are updated using the computed gradients and a learning rate (\(\eta=0.9\)).

**Update Rule:**

```math
\Delta \omega_{ij} = \eta \times \delta_i \times a_j
```

```math
\omega_{ij}^{\text{new}} = \omega_{ij}^{\text{old}} + \Delta \omega_{ij}
```

**Example Updates (see notes):**

- **Weight \(\omega_{16}\) (input 1 → output 6):**

    ```math
    \Delta \omega_{16} = 0.9 \times 0.131 \times 0.392 = 0.046
    ```

    ```math
    \omega_{16}^{\text{new}} = 0.939 - 0.3 = 0.639
    ```

- **Weight \(\omega_{35}\) (input 3 → hidden 5):**

    ```math
    \Delta \omega_{35} = 0.9 \times 0.00653 \times 1 = 0.00588
    ```

    ```math
    \omega_{35}^{\text{new}} = 0.2 - 0.00857 = 0.19143
    ```

---

## 5. Bias Updates

Biases are updated similarly:

```math
b_i^{\text{new}} = b_i^{\text{old}} + \eta \times \delta_i
```

For output neuron 6:

```math
b_6^{\text{new}} = 0.1 + (0.9 \times 0.131) = 0.218
```

![image4](image4)

---

## Key Equations

| Component         | Equation                                              |
| ----------------- | ---------------------------------------------------- |
| Forward Pass      | <span style="font-family:monospace;">\( a_j = \sigma\left(\sum_i \omega_{ij} x_i + b_j\right) \)</span>      |
| Sigmoid           | <span style="font-family:monospace;">\( \sigma(z) = \dfrac{1}{1 + e^{-z}} \)</span>              |
| Output Gradient   | <span style="font-family:monospace;">\( \delta_k = (y - \hat{y}) \cdot \sigma'(o_k) \)</span>   |
| Hidden Gradient   | <span style="font-family:monospace;">\( \delta_j = \sigma'(a_j) \sum_k (\delta_k \omega_{jk}) \)</span> |
| Weight Update     | <span style="font-family:monospace;">\( \Delta \omega_{ij} = \eta \cdot \delta_j \cdot a_i \)</span> |
| Bias Update       | <span style="font-family:monospace;">\( \Delta b_j = \eta \cdot \delta_j \)</span>              |

---

## Why This Matters

Backpropagation efficiently computes gradients by:

1. **Forward pass:** Calculate predictions and cache values.
2. **Backward pass:** Apply the chain rule for error gradients.
3. **Update:** Adjust weights and biases to minimize loss.

Your files and these notes demonstrate a practical implementation of these principles—calculating gradients and updating weights. While the math may appear complex, it systematically optimizes the network using calculus and linear algebra.

---

*Handwritten notes and calculations for visual reference:*

- Network diagram: ![image3](image3)
- Forward pass & error calculation: ![image2](image2)
- Backpropagation and weight update math: ![image1](image1)
- Bias update math: ![image4](image4)

---

**References:**  
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)](http://cs231n.github.io/neural-networks-2/)