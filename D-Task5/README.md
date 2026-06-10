Observation
AND Gate successfully learned.
OR Gate successfully learned.
XOR Gate failed to converge.
Error curve of XOR remained non-zero while AND and OR reached zero error.
What Architectural Change Fixes XOR?

Single-layer perceptron XOR solve nahi kar sakta.

Solution:

Multi-Layer Perceptron (MLP)

Input Layer → Hidden Layer → Output Layer

Hidden layer nonlinear decision boundaries create karti hai jis se XOR correctly classify ho jata hai.

Conclusion

A single-layer perceptron can solve only linearly separable problems such as AND and OR. XOR is not linearly separable, therefore the perceptron fails to converge. Adding one or more hidden layers (Multi-Layer Perceptron) allows the network to learn nonlinear decision boundaries and successfully solve XOR.