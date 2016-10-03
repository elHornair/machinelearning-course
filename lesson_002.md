# Linear regression with multiple variables
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Linear_Regression_with_Multiple_Variables)

## General
* Goal: we try to predict something, based on more than one variable / feature
* Features: x₁, x₂, x₃, ...
* m: amount of examples
* n: amount of features (variables) per example
* xⁱ: input (features) of the i-th training example
* x₂¹: value of the 2-nd feature of the first training example
* x²: n-dimensional vector the second (training example)
* Hypothesis: hθ(x) = θ₀ * x₀ + θ₁ * x₁ + θ₂ * x₂ + θ₃ * x₃ + ...
* We've added x₀ = 1 as the multiplicator for linear factor (for convenience of notation)
* Hypothesis expressed with vectors: hθ(x) = θT*x (the transposed θ-vector multiplied with the
  x-vector)

## Gradient descent for multiple variables
* Hypothesis: hθ(x) = θT*x
* Parameters: θ (vector)
* Cost function: J(θ) = (1/2m) * ∑1..m(hθ(xⁱ) - yⁱ)²
* Gradient descent general: Repeat:**θⱼ := θⱼ - 𝛼 * (d/dθⱼ) * J(θ)** (simultaneously update for
  every j = 0, ..., n)
* Gradient descent concrete: Repeat:**θⱼ := θⱼ - 𝛼 * (1/m) * ∑i=1..m((hθ(xⁱ) - yⁱ) * xⱼⁱ)** (j is the
  index counting the features, i is the index counting the examples for the sum)

## Practical tricks
### Feature Scaling and normalisation
* Make sure that all features are on a similar scale => quicker conversion
* We try to get every feature in the range -1 <= xᵢ <= 1 (1/3 or 3 are still okish, but rather
  not more)
* Formula:**xᵢ = (xᵢ - μᵢ) / sᵢ**
* μ is the average value of xᵢ among all examples -> feature scaling
* s is the range xᵢ among all examples (max value - min value) -> normalisation

### Learning rate
* Make sure that gradient descent is working correctly: plot the value of the cost function J(θ)
  after each iteration. J(θ) should decrease with each iteration
* The plot can also help us judge whether or not the algorithm has converged yet
* Rule of thumb: declare convergence if the value of J(θ) decreases by less than 10⁻³ in one
  iteration
* Plot helps to see errors in gradient descent
* If 𝛼 is too big, so the error becomes bigger instead of smaller with every iteration
* If 𝛼 is too small: slow convergence
* To choose 𝛼, try 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 (each step about 3x the previous value)

## Features and polynomial regression
* Features might be combined. E.g. x₁ = frontage, x₂ = depth. We decide to just use a new feature
  x = x₁ * x₂ (size/area = frontage * depth)
* Polynomial regression: Instead of linear functions for fitting, we can use polynomials. We just
  add more parameters θ to the hypothesis and use the powered version of the feature as new
  features:**hθ(x) = θ₀ + θ₁ * size + θ₂ * size² + θ₃ * size³** (important to apply feature
  scaling here, because the powers make the features a lot bigger)
* We can also use other functions like the square root of a feature, etc. (depending on the data)


# Computing parameters analytically
## Normal equation
* Method for solving for θ analytically
* If θ is a number: J(θ) = a * θ² + b * θ + c => Derivative of the function needs to be zero at the
  local minimum
* If θ is a vector: Same principle, but with all the partial derivatives set to 0
* How to do this: pack all the features values (columns) of all the examples (rows) in a matrix *X*
  (set 1 for x₀). You get a m * (n + 1) matrix. Pack all y-values in a y-vector of dimension m
* X is called the**design matrix**
* Solution is:**θ = (XT * X)⁻¹ * XT * y**
* Octave: pinv(X' * X) * X' * y
* For small sets of features, this is appropriate and simple (because no iterations, no need to
  choose 𝛼). For large sets of features (n > 1000 or 10'000), it becomes really slow, because
  calculating the inverse matrix has a complexity of O(n³)
* Noninvertibility:
    * Some matrices can't be inverted - mathematically troublesome, but Octave takes
      care of this (as long as we use the pseudo-inverse-function*pinv*instead of*inv*)
    * Reasons:
        * Redundant features (linearly dependent), e.g. the size in feet and in m³
        * Too many features or not enough training data (n >= m). In this case we can delete some
          features or us regularisation (will be covered later)


 # Octave
 Octave Documentation: [Octave Documentation](https://www.gnu.org/software/octave/doc/interpreter/)
 Octave Tutorial: [Octave Tutorial](https://share.coursera.org/wiki/index.php/ML:Octave_Tutorial)


##  Quiz:
### Question 1
      x = (x - avg) / range
     
      x = 94
      avg: (89 + 72 + 94 + 69) / 4 = 324 / 4 = 81
      range: 94 - 69 = 25
     
      x = (94 - 81) / 25 = 0.52


##  Programming assignment:
### How to calculate the cost function
* J = 1/(2 * m) * sum((X * theta - y).^2)

### Gradient descent as matrix operations
* As we need to multiply the elements of each row of "X * theta - y" with their respective column
  in X (the "inner loop" is for the examples), we just multiply everything with the transpose of X.
  This takes care of building the sum at the same time
* theta = theta - alpha * (1/m) * X' * (X * theta - y)
