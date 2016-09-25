# Introduction
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Introduction)

## Supervised learning
* Goal: inferring (ableiten) a function from labeled**training data**
* The training data consist of a set of training examples
* Each example is a pair consisting of an input object (typically a vector) and a desired output value
* Each data element tells us, what is it's**right answer** in this problem
* Two types:**regression problem**and**classification problem**

### Regression problem
* Example: We have data set of houses with price and size in m^3. We try to find a function that can
  predict the price of a certain house, when given iz size in m^3.
* Mathematically, we're trying to fit a line through the given data that's as appropriate as possible

### Classification problem
* Example: We have a data set of women with breast cancer. For each sample, we know the age of the
  woman, the size of the cancer and if the cancer was malignant (bösartig) or benign (freundlich).
  We are now given a new data pair of age and size. For this pair, we try to classify, if the cancer
  is malignant or benign.
* Mathematically, we're trying to predict a discrete value output, based on the given data
* The output can be binary (true/false), but can also have more (but always discrete) possible outputs
  like "Type A", "Type B", "Type C" and "Type D" for example.

## Unsupervised learning
* Goal: draw inferences (Folgen) from datasets consisting of input data without labeled responses
* Common: cluster analysis - For finding hidden patterns or grouping in data
* The data elements don't tell us, what their**right answer** in this problem is
* The algorithm has to find the structure in the data by itself and break the data into clusters
* Example: Google news groups news stories from different sources into "the same story"
* Example: market segmentation (algorithm divides customers into groups with similar attributes)


# Linear Regression with one variable
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Linear_Regression_with_One_Variable)

## Model representation
* Training example: pair of x and y
* x: input variable (features)
* y: output variables / target variable (the value we're trying to predict)
* m: the amount of training examples
* Goal of the learning algorithm: output a**hypothesis function h(x)** for predicting y based
  on x
* h(x) is a linear (thus univariate, one variable!) function of the form:
  **hθ(x) = θ₀ + θ₁ * x**(θ: greek Theta)
* θ₀ and θ₁: parameters of the hypothesis

## Cost function
* How should we choose θ₀ and θ₁ so our hypothesis is as appropriate as possible?
* Idea: choose θ₀ and θ₁ so that hθ(x) is close to y for our training examples
* Goal:**minimize cost function J(θ₀,θ₁) = (1/2m) * ∑1..m(hθ(xᵢ) - yᵢ)²** (we prepend
  1/m so we get the average error; 1/2 is only prepended for simplifying the math)
* We can choose θ₀ and θ₁ freely for making (1/2m) * ∑(hθ(xᵢ) - yᵢ)² as small as possible
* This cost function is called the **squared error cost function** (there are other cost functions
  but this one performs well for most linear regression problems)
* Two levels of abstraction:
  * The hypothesis hθ(x) is a function of x (because θ is fixed in one particular hypothesis)
  * The cost function J(θ₀,θ₁) is a function of θ₀ and θ₁


# Gradient descent
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Gradient_Descent)

## General
* Algorithm for optimisation (minimisation) problems
* Idea:
  1. Start at some point in the plot
  2. Go in the direction with the biggest descent
  3. repeat step 2 until local minimum is found (if we have two parameters, the local minimum is
     also the global minimum)
* Reminder: the level of descent is mathematically expressed with the derivative dx
* Mathematically:
  1. θ₀ = 0; θ₁ = 0
  2. **θⱼ = θⱼ -  𝛼 * (d/dθⱼ) * J(θ₀,θ₁)** (general)
    * θ₀ = θ₀ -  𝛼 * (d/dθ₀) * J(θ₀,θ₁) (specific for θ₀)
    * θ₁ = θ₁ -  𝛼 * (d/dθ₁) * J(θ₀,θ₁) (specific for θ₁)
    * Important: simultaneous update (don't use the already updated θ₀ for calculating θ₁ in the same iteration)
    * 𝛼: learning rate (determines, how big a step we take)
    * (d/dθⱼ) * J(θ₀,θ₁): derivative term
* If 𝛼 is too small, we're just taking baby steps until convergence
* If 𝛼 is too big, we pass the optimum over and over again (going back and forth)
* At the local minimum, the derivative term will be 0, so θⱼ will not change anymore
* 𝛼 can be fixed, because the derivative term gets smaller the closer we get to the local minimum

## Gradient descent for linear regression
* We apply the gradient descent algorithm for minimising the cost function
* Derivative term in this case: (d/dθⱼ) * (1/2m) * ∑1..m(hθ(xᵢ) - yᵢ)² =
  (d/dθⱼ) * (1/2m) * ∑1..m(θ₀ + θ₁ * xᵢ - yᵢ)²
*  Specific partial derivative terms (the math was done with some calculus magic):
    * (d/dθ₀) * (1/2m) * ∑1..m(hθ(xᵢ) - yᵢ)² = (1/m) * ∑1..m(hθ(xᵢ) - yᵢ)
    * (d/dθ₁) * (1/2m) * ∑1..m(hθ(xᵢ) - yᵢ)² = (1/m) * ∑1..m(hθ(xᵢ) - yᵢ) * xᵢ
* Specific algorithm (remember to make simultaneous update):
     * θ₀ = θ₀ - 𝛼 * (1/m) * ∑1..m(hθ(xᵢ) - yᵢ)
     * θ₁ = θ₁ - 𝛼 * (1/m) * ∑1..m(hθ(xᵢ) - yᵢ) * xᵢ
* Side note: this is called the**batch gradient descent** algorithm (because we use all data
  samples for each step)

##  Quiz:
### Question 2
      (1/2m) * ∑1..m(hθ(xᵢ) - yᵢ)²
    = (1/2m) * ∑1..m(θ₀ + θ₁ * xᵢ - yᵢ)²
    = (1/2m) * ∑1..m(0 + 1 * xᵢ - yᵢ)²
    = (1/2m) * ∑1..m(xᵢ - yᵢ)²
    = (1/(2*4)) * ∑1..4(xᵢ - yᵢ)²
    = (1/8) * ∑1..4(xᵢ - yᵢ)²
    = (1/8) * ((3 - 4)² + (2 - 1)² + (4 - 3)² + (0 - 1)²)
    = (1/8) * ((-1)² + (1)² + (1)² + (-1)²)
    = (1/8) * (1 + 1 + 1 + 1)
    = (1/8) * (4)
    = (4/8)
    = 1/2
    = 0.5

### Question 3
    hθ(x) = θ₀ + θ₁ * x
    hθ(x) = -1 + 2 * x
    hθ(6) = -1 + 2 * 6
    hθ(6) = -1 + 12
    hθ(6) = 11


# Linear algebra review
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Linear_Algebra_Review)

## Matrix
* Two dimensional array
* Dimension of matrix: number of rows * number of columns
*     # Example (3*2 matrix => matrix is an element of the set R3*2)
      A = |122 877|
          | 12 112|
          |122   3|
* Matrix elements: Aᵢⱼ refers to the element in the  i-th row in the j-th column of A (1-indexed)

## Vector
* A matrix with just one column (a n*1 matrix)
*     # Example (4-dimensional vector)
      y = |122|
          | 12|
          |122|
          |  3|
* yᵢ refers to the i-th element of y (1-indexed, but sometimes also 0-indexed)

## Addition and scalar multiplication
