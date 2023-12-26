*`In this we are going to cover some of the most fundamental concepts of tensor using TensorFlow More specifically, we are going to cover:`* 

- `I*ntroduction to tensors*`
- *`Getting information from tensors`*
- *`Manipulating tensors`*
- *`Tensors & Numpy`*
- *`Using @tf.functions ( a way to speed up your regular python functions)`*
- *`Using GPUs with Tensorflow or TPUs`*

`tf.constant` is a function in TensorFlow used to create a tensor with a constant value. The syntax is `tf.constant(value, dtype=None, shape=None, name='Const')`.

- `value` is an actual constant value which will be used in the tensor
- `dtype` is the data type of elements stored in the tensor
- `shape` represents the shape of the tensor
- `name` is the name for the tensor

This function returns a constant tensor, and the elements of the resulting tensor are all the same as the `value` provided. This is useful when you need a tensor of a specific shape and size filled with a constant value.

```python
# create tensor with tf.constant()
scalar = tf.constant(7)
print(scalar)
scalar.ndim
```

The `.ndim` attribute of a tensor gives us the number of dimensions the tensor has. In the context of tensors, a dimension refers to the degree of freedom available for any operation. For a scalar, `.ndim` would return 0, as it has no other dimensions but the value itself. It has no length, width, or depth. It's a single number. For example, if you run `scalar.ndim` after the above code, it will output 0, indicating that our tensor, `scalar`, is a 0-dimensional tensor, or in other words, a scalar.

1. For the first example:

```python
vector = tf.constant([[10, 10, 10, 10], [10, 10, 10, 10]])
vector.ndim

```

Here, `vector` is a 2D tensor (matrix) with dimensions 2x4, so `vector.ndim` will be 2.

1. For the second example:

```python
vector = tf.constant([[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]])
vector.ndim

```

In this case, `vector` is a 2D tensor with dimensions 3x4, so `vector.ndim` will still be 2. The number of elements in each inner list (4 elements) determines the second dimension, and the number of inner lists (3 lists) determines the first dimension.

In both examples, the `ndim` is 2, indicating that the tensors are 2-dimensional.

The provided `vector` is indeed a 3D tensor. Let's break down the structure:

```python
vector = tf.constant([[[10, 10, 10, 10], [10, 10, 10, 10]],
                     [[10, 10, 10, 10], [10, 10, 10, 10]]])

```

- The outermost list contains two elements, making it the first dimension.
- Each of these elements is a list, representing the second dimension.
- Inside these second-level lists, there are two more lists, forming the third dimension.
- Finally, within the innermost lists, you have individual elements (numbers), which make up the fourth dimension (though in this case, it's just scalar values).

So, the overall shape of this 3D tensor is (2, 2, 4), representing (first dimension, second dimension, third dimension).

In a real-world context, you can think of a 3D tensor like this as representing a cube of data. Let's use a metaphor for spatial coordinates:

1. **First Dimension (Z-Axis):**
    - The outermost list `[...]` represents the first dimension or the Z-axis. Imagine it as the height of a cube.
2. **Second Dimension (Y-Axis):**
    - Each element within the outer list (e.g., `[[...], [...]]`) represents the second dimension or the Y-axis. Think of these as stacked layers within the cube.
3. **Third Dimension (X-Axis):**
    - The innermost lists (e.g., `[[10, 10, 10, 10], [10, 10, 10, 10]]`) represent the third dimension or the X-axis. Visualize these as rows within each layer.

Each element within the innermost lists corresponds to a data point or a value within the cube. In terms of coordinates, you can interpret it like this:

- `(0, 0, 0)`: The first element in the first row of the first layer.
- `(0, 0, 1)`: The second element in the first row of the first layer.
- `(0, 1, 0)`: The first element in the second row of the first layer.
- `(1, 0, 0)`: The first element in the first row of the second layer.

So, in a real-world graph, you can visualize it as a cube where each element or number is located at a specific position within that cube, defined by the three dimensions.

Vectors and matrices are both types of mathematical objects used in linear algebra, but they have distinct differences in terms of their dimensions and operations.

### Vectors:

1. **Dimension:**
    - A vector is a one-dimensional array of numbers. It can be represented as a column or row.
2. **Notation:**
    

![image](https://github.com/SequenceAI/understanding_deeplearning/assets/21698209/f450c040-b639-46e0-9a2a-ed44b0ddd3f1)



    

1. **Size:**
    - The size of a vector is determined by the number of elements it contains.
2. **Operations:**
    - Common operations include addition, subtraction, scalar multiplication, dot product, and vector norms.

### Matrices:

1. **Dimension:**
    - A matrix is a two-dimensional array of numbers. It has rows and columns.
2. **Notation:**
    
 
![image](https://github.com/SequenceAI/understanding_deeplearning/assets/21698209/6c226458-9462-4da5-851a-6af0514a33c3)
    
3. **Size:**
    - The size of a matrix is given by the number of rows and columns it has.
4. **Operations:**
    - Common operations include addition, subtraction, scalar multiplication, matrix multiplication, and determinant computation.

### Key Differences:

1. **Dimensions:**
    - Vectors have one dimension (either a row or a column).
    - Matrices have two dimensions (rows and columns).
2. **Notation:**
    - Vectors can be represented as a single column or row.
    - Matrices are represented as a grid of numbers with rows and columns.
3. **Size:**
    - The size of a vector is determined by the number of elements.
    - The size of a matrix is given by the number of rows and columns.
4. **Applications:**
    - Vectors are often used to represent quantities like forces, velocities, or coordinates.
    - Matrices are used to represent transformations, systems of linear equations, and other structured data.

In summary, vectors and matrices are both fundamental structures in linear algebra, but vectors are one-dimensional arrays, while matrices are two-dimensional arrays. The operations and applications associated with each depend on their respective dimensions and structures.

### Tensors:

- **Dimension:**
    - Tensors are a generalization of vectors and matrices and can have more than two dimensions.
    - Scalars (single numbers) and vectors are 0-D and 1-D tensors, respectively. Matrices are 2-D tensors.
- **Notation:**
    - Higher-dimensional arrays.
    - Examples:
        
  ![image](https://github.com/SequenceAI/understanding_deeplearning/assets/21698209/43332558-feb8-4969-8fb3-d3d0c99fa0e0)

        
    

### Relationship:

- Vectors and matrices are specific cases of tensors.
- A vector is a 1-D tensor, and a matrix is a 2-D tensor.
- Tensors can have any number of dimensions, making them more flexible for representing complex data structures.

### Applications:

- **Vectors:** Represent quantities like forces, velocities, or coordinates.
- **Matrices:** Used for transformations, systems of linear equations, and structured data.
- **Tensors:** Applied in deep learning and data science to represent multi-dimensional data, such as images, videos, and higher-order data structures.

In summary, while vectors and matrices are specific cases of tensors, tensors provide a more general framework for handling multi-dimensional data, making them versatile for various applications, especially in fields like deep learning and scientific computing.

`tf.Variable` is a class in TensorFlow used for representing a mutable tensor whose value can be changed. This mutable state is helpful especially in machine learning applications where we need to change the values of weights and biases in a neural network during training.

When you define a variable, you must provide an initial value. TensorFlow uses the data type of this initial value to infer the data type of the variable. For example:

```python
# Create a variable
initial_value = tf.constant([1, 2])
a = tf.Variable(initial_value)
print(a)

```

In the above example, `initial_value` is a constant tensor with values `[1, 2]`. We use this to create a variable `a`. TensorFlow will infer the data type of `a` from `initial_value`, which is `int32` in this case.

Variables in TensorFlow can be updated and retrieved using built-in methods. You can use the `.assign(value)`, `.assign_add(increment)`, and `.assign_sub(decrement)` methods to update the value of a variable:

```python
# Update the value of a variable
a.assign([2, 3])
print(a)

# Add a value to a variable
a.assign_add([2, 3])
print(a)

# Subtract a value from a variable
a.assign_sub([2, 3])
print(a)

```

In the above example, `.assign([2, 3])` updates the value of `a` to `[2, 3]`. `.assign_add([2, 3])` adds `[2, 3]` to the current value of `a`. `.assign_sub([2, 3])` subtracts `[2, 3]` from the current value of `a`.

Remember that these operations change the value of the variable in-place. That is, they modify the actual values of the variable rather than creating a new tensor.

In summary, `tf.Variable` allows you to maintain and update mutable tensor-like values in your TensorFlow computations, making it especially useful for things like model parameters in machine learning.

When it's mentioned that TensorFlow variables cannot be reshaped, it refers to the fact that you cannot directly change the shape of a TensorFlow variable after it has been created.

In TensorFlow, variables are special tensors that are used to hold and update parameters in a model during training. Once you create a variable with a specific shape, you typically cannot change that shape later on. Reshaping involves changing the dimensions (number of elements along each axis) of a tensor without changing its data. For regular tensors (not variables), you can use `tf.reshape` to reshape them, but this operation doesn't work for variables.

Here's an example to illustrate:

```python
import tensorflow as tf

# Creating a variable with shape (2, 3)
my_variable = tf.Variable([[1, 2, 3], [4, 5, 6]])

# Attempting to reshape the variable
try:
    reshaped_variable = tf.reshape(my_variable, (3, 2))
except Exception as e:
    print(f"Error: {e}")

```

This would raise an error like: "TypeError: Expected binary or unicode string, got 3". TensorFlow expects a constant or tensor-like object for the new shape argument, and reshaping a variable directly like this is not supported.

If you need to change the shape of a variable, one common approach is to create a new variable with the desired shape and initialize it with the values from the original variable. TensorFlow provides various ways to initialize variables and assign values, but direct reshaping is not one of them for variables.

Reshaping refers to the process of changing the shape of a tensor, which involves rearranging its elements to have a different number of dimensions or a different size along each dimension. In the context of neural networks and machine learning frameworks like TensorFlow, reshaping is a common operation performed on tensors.

For example, if you have a tensor with shape (3, 4), it means it has 3 rows and 4 columns. Reshaping this tensor into a shape of (2, 6) would rearrange the same set of elements into a new structure with 2 rows and 6 columns. The total number of elements remains the same; only the arrangement of those elements changes.

Here's a simple example in Python using NumPy:

```python
import numpy as np

# Create a 3x4 matrix
original_matrix = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]])

# Reshape it into a 2x6 matrix
reshaped_matrix = original_matrix.reshape((2, 6))

print("Original matrix:")
print(original_matrix)
print("\\nReshaped matrix:")
print(reshaped_matrix)

```

In this case, the elements of the original 3x4 matrix are rearranged to form a new 2x6 matrix:

```
Original matrix:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

Reshaped matrix:
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]

```

Reshaping is a fundamental operation in deep learning and neural networks, where it's often used to prepare data for input to layers or to adapt the shape of tensors during the construction of a neural network model.
<<<<<<< HEAD
=======


```python
random_tensor_1 = tf.random.uniform(shape=(3,2))
random_tensor_2 = tf.random.Generator.from_seed(42)
random_tensor_2.uniform((3,2))
random_tensor_2.normal((3,2))

```

The code you provided involves generating random tensors using TensorFlow's random module. Let's break down the differences between `tf.random.uniform` and `tf.random.Generator`:

1. **`tf.random.uniform(shape=(3,2))`**:
   - This creates a random tensor with a shape of (3, 2) using a default random number generator.
   - The values in the tensor are drawn from a uniform distribution between 0 (inclusive) and 1 (exclusive).

```python
random_tensor_1 = tf.random.uniform(shape=(3, 2))
```

2. **`tf.random.Generator.from_seed(42)`**:
   - This creates a random number generator with a specific seed (in this case, 42) using the `tf.random.Generator` class.
   - Setting a seed ensures reproducibility, meaning if you run the same code with the same seed, you'll get the same sequence of random numbers.

```python
random_tensor_2 = tf.random.Generator.from_seed(42)
```

3. **`random_tensor_2.uniform((3,2))`**:
   - This generates a random tensor with a shape of (3, 2) using the random number generator (`random_tensor_2`) created above.
   - The values are drawn from a uniform distribution between 0 (inclusive) and 1 (exclusive).

```python
random_uniform_tensor = random_tensor_2.uniform((3, 2))
```

4. **`random_tensor_2.normal((3,2))`**:
   - This generates a random tensor with a shape of (3, 2) using the same random number generator (`random_tensor_2`).
   - The values are drawn from a normal (Gaussian) distribution with mean 0 and standard deviation 1.

```python
random_normal_tensor = random_tensor_2.normal((3, 2))
```

In summary, `tf.random.uniform` is a convenience function for generating random tensors directly, while `tf.random.Generator` allows you to create a random number generator with a specific seed and then use it to generate random tensors with various distributions, such as uniform or normal. The latter is often preferred when you need reproducibility in your random experiments.

In the context of random number generation, a "seed" is a starting point or initial value used to initialize a random number generator. The idea is that if you use the same seed, you will get the same sequence of random numbers each time you run your program. This is particularly useful when you want to ensure reproducibility in your experiments or simulations.

Here's how it works:

1. **Seed Initialization**: When you set a seed for a random number generator, it initializes its internal state based on that seed.

2. **Deterministic Sequence**: Once the generator is initialized with a seed, it follows a deterministic algorithm to produce a sequence of seemingly random numbers. The sequence appears random, but it's entirely determined by the initial seed.

3. **Reproducibility**: If you use the same seed in different runs of your program or script, you'll get the exact same sequence of random numbers. This is crucial when you want to reproduce experimental results or share code with others, ensuring that everyone gets the same random data.

In the case of `tf.random.Generator.from_seed(42)`, the seed is set to the integer value 42. You can choose any integer value as a seed, and as long as you use the same seed, you'll get the same sequence of random numbers.

Here's an example:

```python
import tensorflow as tf

# Create a random number generator with seed 42
rng = tf.random.Generator.from_seed(42)

# Generate some random numbers
random_numbers_1 = rng.uniform((3, 3))
print(random_numbers_1)

# Re-create the random number generator with the same seed
rng = tf.random.Generator.from_seed(42)

# Generate the same random numbers again
random_numbers_2 = rng.uniform((3, 3))
print(random_numbers_2)
```

In this example, `random_numbers_1` and `random_numbers_2` will be identical because the same seed (42) was used to initialize the random number generator in both cases.


# Uniform Distributions vs Normal Distributions

1. **Normal Distribution (Gaussian Distribution):**
    - **Shape:** Bell-shaped curve.
    - **Central Tendency:** Symmetrical around the mean.
    - **Probability Density:** Higher probability of values near the mean, decreasing as you move away.
    - **Examples:** Physical measurements like height, weight, errors in measurements.
    - **Probability Density Function (PDF):** Describes the likelihood of different outcomes.
    
    !https://u![img.png](img.png)pload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/700px-Normal_Distribution_PDF.svg.png
    
2. **Uniform Distribution:**
    - **Shape:** Constant probability across the range.
    - **Central Tendency:** No specific point of concentration.
    - **Probability Density:** Equal probability for all values in the range.
    - **Examples:** Rolling a fair die, selecting a random number from a given range.
    - **Probability Density Function (PDF):** Flat and constant.
    
    ![img_1.png](img_1.png)

In a normal distribution, most values cluster around the mean, with fewer values as you move away from the mean. In a uniform distribution, all values have an equal probability of occurring. These distributions are fundamental in statistics and probability theory and are used to model and understand various phenomena in the real world.

In TensorFlow, both `tf.random.Generator.from_seed(seed)` and `tf.random.set_seed(seed)` are used to set the seed for reproducibility in random number generation, but they operate in different ways.

1. **`tf.random.Generator.from_seed(seed)`**:
   - This method creates a new random number generator instance with the specified seed.
   - The generator instance can then be used to generate random numbers with various distributions.
   - If you create multiple generator instances with the same seed, they will produce independent sequences of random numbers.

   Example:
   ```python
   rng = tf.random.Generator.from_seed(42)
   random_numbers = rng.normal(shape=(3, 2))
   ```

2. **`tf.random.set_seed(seed)`**:
   - This function sets the global seed for the TensorFlow random number generator.
   - It affects all random number generators created afterwards in the TensorFlow session.
   - It's a simpler way to set the seed globally for the entire TensorFlow environment.

   Example:
   ```python
   tf.random.set_seed(42)
   random_numbers = tf.random.normal(shape=(3, 2))
   ```

In summary, `tf.random.Generator.from_seed(seed)` is useful when you want to create independent random number generators with specific seeds, while `tf.random.set_seed(seed)` sets the seed globally for the entire TensorFlow environment and affects all random operations. You can choose the method that best fits your needs for reproducibility.

![img_2.png](img_2.png)
