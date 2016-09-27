# Linear algebra review
Wiki: [Wiki](https://share.coursera.org/wiki/index.php/ML:Linear_Algebra_Review)

## Matrix
* Two dimensional array
* Dimension of matrix: number of rows * number of columns
*     # Example (3*2 matrix => matrix is an element of the set R3*2)
      A = |122  877|
          | 12  112|
          |122    3|
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
### Addition
* Add up each element of matrix A with it's corresponding element in matrix B
* Both matrices need to have the same dimension
* The resulting matrix has the same dimension as the two input matrices
*     # Example
      A = | 12  1|
          |312  2|
          |  3  3|
          
      B = |  4  1|
          |  1  2|
          |100  3|
          
      A + B = | (12 + 4)  (1+1)| = | 16  2|
              |(312 + 1)  (2+2)|   |313  4|
              |(3 + 100)  (3+3)|   |103  9|

### Scalar multiplication
* Multiply each element with the scalar
* The resulting matrix has the same dimension as the input matrix
*     # Example
      A = | 12  1|
          |312  2|
          |  3  3|
          
      2 * A = | 12  1| = |(2 * 12)   (2 * 1)| = | 24  2|
              |312  2|   |(2 * 312)  (2 * 2)|   |624  4|
              |  3  3|   |(2 * 3)    (2 * 3)|   |  6  6|
          
## Matrix vector multiplication
* The columns of the matrix must match the dimension of the vector
* Multiply the elements of each row with the elements of the vector and add them up
* For**A * x = y** we have to multiply the elements of A's i-th row with the elements of x for
  calculating yᵢ
* A m * n matrix, multiplied with a n-dimensional (n * 1) vector results in a m-dimensional vector
*     # Example
      A = | 12  1|
          |312  2|
          |  3  3|
          
      x = | 2|
          |10|
          
      A * x = | (12 * 2) + (1 * 10)| = | (24) + (10)| = | 34|
              |(312 * 2) + (2 * 10)|   |(624) + (20)|   |644|
              |  (3 * 2) + (3 * 10)|   |  (6) + (30)|   | 36|
* We can use this**for solving a equation for each example in a big dataset**(f.e. calculating hθ(x)
  for each example in the training data for a regression problem)
*     # Example
      # House sizes: 2104, 1416, 1534, 852
      # hθ(x): -40 + 0.25x
          
      A = |1  2104| # data matrix
          |1  1416|
          |1  1534|
          |1   852|
          
      b = | -40| # vector with the parameters
          |0.25|
          
      A * b = | -40 + 0.25 * 2140| = |hθ(2140)| # prediction
              | -40 + 0.25 * 1416|   |hθ(1416)|
              | -40 + 0.25 * 1534|   |hθ(1534)|
              | -40 + 0.25 *  852|   |hθ( 852)|

## Matrix matrix multiplication
* The columns of the first matrix must match the rows of the second matrix
* Multiply the elements of each row of the matrix with the elements of the corresponding column
  of the second matrix and add them up
* For**A * B = C** we have to multiply A with the i-th column of B for calculating the i-th
  column of C
* A m * n matrix, multiplied with a n * o matrix results in a m * o matrix
*     # Example
      A = |1  3|
          |2  5|
       
      B = |0  1|
          |3  2|
          
      A * B = |1  3| * |0  1| = |(1 * 0) + (3 * 3)   (1 * 1) + (3 * 2)| = | 9    7|
              |2  5|   |3  2|   |(2 * 0) + (5 * 3)   (2 * 1) + (5 * 2)|   |15   12|
* We can use this**for solving many equations for each example in a big dataset**(f.e.
  calculating many hθ(x) for each example in the training data for a regression problem)
*     # Example
      # House sizes: 2104, 1416, 1534, 852
      # hθ₁(x): -40 + 0.25x
      # hθ₂(x): 200 + 0.10x
          
      A = |1  2104| # data matrix
          |1  1416|
          |1  1534|
          |1   852|
          
      B = | -40   200| # matrix with the vectors of parameters
          |0.25   0.1|
          
      A * B = |hθ₁(2140)   hθ₁(2140)| # prediction
              |hθ₁(1416)   hθ₂(1416)|
              |hθ₁(1534)   hθ₂(1534)|
              |hθ₁( 852)   hθ₂( 852)|

## Matrix multiplication properties and special matrix operations
* **Not commutative**-> A * B != B * A
* **Associative**-> (A * B) * C == A * (B * C)

### Identity matrix
* The matrix *I* we can multiply a matrix *A* with and the result will be the
  *A* again
* There is a different identity matrix for each dimension (the n*n identity matrix)
* Identity matrices always have 1s along the diagonal axis and 0s at everywhere else
*     # Examples
      |1  0| # 2*2 identity matrix
      |0  1|
       
      |1  0  0| # 3*3 identity matrix
      |0  1  0|
      |0  0  1|
* For any matrix A:**A * I = I * A = A**

## Inverse matrix
* Symbol: *A⁻¹*
* If we multiply a matrix with it's inverse matrix, we get the identity matrix
* If *A* is a *m * m* matrix and if it has an inverse, then **A * A⁻¹ = A⁻¹ * A = I**
* So only square matrices have inverses
* Calculating the inverse matrix: not very easy by hand, so we just use helper functions :)
* Some matrices don't have an inverse (e.g. the matrix that's all 0s). They are called
  *singular* or *degenerate* matrices

## Matrix transponse
* The rows become the columns, the columns become the rows
*     # Example
      A = |1  2  0|
          |3  5  9|
       
      AT = |1  3|
           |2  5|
           |0  9|
* Formal: let *A* be an *m * n* matrix and *B = AT*. Then *B* is an *n * m* matrix and
  *Bᵢⱼ = Aⱼᵢ*
