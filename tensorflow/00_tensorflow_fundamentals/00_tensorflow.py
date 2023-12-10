'''
    In this we are going to cover some of the most fundamental concepts of tensor using TensorFlow
    More specifically, we are going to cover:
    - introduction to tensors
    - Getting information from tensors
    - Manipulating tensors
    - Tensors & Numpy
    - Using @tf.functions ( a way to speed up your regular python functions)
    - Using GPUs with Tensorflow or TPUs

'''
# Import Tensorflow
import tensorflow as tf
print(tf.__version__)

# create tensor with tf.constant()
scalar = tf.constant(7)
scalar

# check the number of dimenstions of a tensor (ndim stands for number of dimenstions)
scalar.ndim


# create a 3 dimension
outer_list = []
middle_list = []
inner_list = [10, 10, 10, 10]
for i in range(0,4):
    middle_list.append(inner_list)
outer_list.append(middle_list)



# create vector
vector = tf.constant([10, 7])
vector.ndim

# crate a matrix (has more than 1 dimension)
matrix = tf.constant([[10, 7],[7, 10]])
matrix.ndim

# create another matrix
another_matrix = tf.constant([[10., 7.],
                              [7., 10.],
                              [8., 9.]], dtype=tf.float16)

another_matrix

''' Creating Tensor with tf.Variables'''
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])

pass_in_unchangeable = tf.Variable(unchangeable_tensor)


''' Let's try to change one of the elements in our changeable tensor'''
changeable_tensor[0].assign(7)
changeable_tensor.assign([11, 11],True)

