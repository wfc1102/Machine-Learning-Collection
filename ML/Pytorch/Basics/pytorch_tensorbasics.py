"""
Walk through of a lot of different useful Tensor Operations, where we
go through what I think are four main parts in:

1. Initialization of a Tensor
2. Tensor Mathematical Operations and Comparison
3. Tensor Indexing
4. Tensor Reshaping

But also other things such as setting the device (GPU/CPU) and converting
between different types (int, float etc) and how to convert a tensor to an
numpy array and vice-versa.

Programmed by Aladdin Persson
* 2020-06-27: Initial coding
* 2022-12-19: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import numpy as np

# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!
print(f"\"cuda\" if torch.cuda.is_available() else \"cpu\": {device}")

# Initializing a Tensor in this case of shape 2x3 (2 rows, 3 columns)
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
print(f"torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True): \n{my_tensor}\n")

# A few tensor attributes
print(f"my_tensor.dtype : {my_tensor.dtype}")  # Prints dtype of the tensor (torch.float32, etc)
print(f"my_tensor.device: {my_tensor.device}")  # Prints cpu/cuda (followed by gpu number)
print(f"my_tensor.shape: {my_tensor.shape}")  # Prints shape, in this case 2x3
print(f"my_tensor.requires_grad: {my_tensor.requires_grad}")  # Prints true/false

# Other common initialization methods (there exists a ton more)
x = torch.empty(size=(3, 3))  # Tensor of shape 3x3 with uninitialized data
print(f"x = torch.empty(size=(3, 3)): \n{x}\n")
x = torch.zeros((3, 3))  # Tensor of shape 3x3 with values of 0
print(f"x = torch.zeros((3, 3)): \n{x}\n")
x = torch.rand((3, 3))  # Tensor of shape 3x3 with values from uniform distribution in interval [0,1)
print(f"x = torch.rand((3, 3)): \n{x}\n")
x = torch.ones((3, 3))  # Tensor of shape 3x3 with values of 1
print(f"x = torch.ones((3, 3)): \n{x}\n")
x = torch.eye(5, 5)  # Returns Identity Matrix I, (I <-> Eye), matrix of shape 2x3
print(f"x = torch.eye(5, 5): \n{x}\n")
x = torch.arange(start=0, end=5, step=1) # Tensor [0, 1, 2, 3, 4], note, can also do: torch.arange(11)
print(f"x = torch.arange(start=0, end=5, step=1): \n{x}\n")
x = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
print(f"x = torch.linspace(start=0.1, end=1, steps=10): \n{x}\n")
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # Normally distributed with mean=0, std=1
print(f"x = torch.empty(size=(1, 5)).normal_(mean=0, std=1): \n{x}\n")
x = torch.empty(size=(1, 5)).uniform_(0, 1)  # Values from a uniform distribution low=0, high=1
print(f"x = torch.empty(size=(1, 5)).uniform_(0, 1): \n{x}\n")
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3
print(f"x = torch.diag(torch.ones(3)): \n{x}\n")

# How to make initialized tensors to other types (int, float, double)
# These will work even if you're on CPU or CUDA!
tensor = torch.arange(4)  # [0, 1, 2, 3] Initialized as int64 by default
print(f"tensor = torch.arange(4): \n{tensor}\n")
print(f"tensor.bool(): \n{tensor.bool()}\n")  # Converted to Boolean: 1 if nonzero
print(f"tensor.short(): \n{tensor.short()}\n")  # Converted to int16
print(f"tensor.long(): \n{tensor.long()}\n")  # Converted to int64 (This one is very important, used super often)
print(f"tensor.half():  \n{tensor.half()}\n")  # Converted to float16
print(f"tensor.float(): \n{tensor.float()}\n")  # Converted to float32 (This one is very important, used super often)
print(f"tensor.double(): \n{tensor.double()}\n")  # Converted to float64

# Array to Tensor conversion and vice-versa
np_array = np.zeros((5, 5))
print(f"np_array = np.zeros((5, 5)): \n{np_array}\n")  # Converted to float64
tensor = torch.from_numpy(np_array)
print(f"tensor = torch.from_numpy(np_array): \n{tensor}\n")  # Converted to float64
np_array_again = tensor.numpy()  # np_array_again will be same as np_array (perhaps with numerical round offs)
print(f"np_array_again = tensor.numpy(): \n{np_array_again}\n")  # Converted to float64

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
print(f"x = torch.tensor([1, 2, 3]): \n{x}\n")  # Converted to float64
y = torch.tensor([9, 8, 7])
print(f"y = torch.tensor([9, 8, 7]): \n{y}\n")  # Converted to float64

# -- Addition --
z1 = torch.empty(3)
print(f"z1 = torch.empty(3): \n{z1}\n")  # Converted to float64
torch.add(x, y, out=z1)  # This is one way
print(f"torch.add(x, y, out=z1): \n{z1}\n")  # Converted to float64
z2 = torch.add(x, y)  # This is another way
print(f"z2 = torch.add(x, y): \n{z2}\n")  # Converted to float64
z = x + y  # This is my preferred way, simple and clean.
print(f"z = x + y: \n{z}\n")  # Converted to float64

# -- Subtraction --
z = x - y  # We can do similarly as the preferred way of addition
print(f"z = x - y: \n{z}\n")  # Converted to float64

# -- Division (A bit clunky) --
z = torch.true_divide(x, y)  # Will do element wise division if of equal shape
print(f"z = torch.true_divide(x, y): \n{z}\n")  # Converted to float64

# -- Inplace Operations --
# -- DO NOT RECOMMAND unless memory is small: https://www.jb51.net/python/294076bz9.htm
t = torch.zeros(3)
print(f"t = torch.zeros(3): \n{t}\n")  # Converted to float64
t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
print(f"t.add_(x): \n{t}\n")
t += x  # Also inplace: t = t + x is not inplace
print(f"t += x: \n{t}\n")  # Converted to float64

# -- Exponentiation (Element wise if vector or matrices) --
z = x.pow(2)  # z = [1, 4, 9]
print(f"z = x.pow(2): \n{z}\n")
z = x**2  # z = [1, 4, 9]
print(f"z = x**2: \n{z}\n")

# -- Simple Comparison --
z = x > 0  # Returns [True, True, True]
print(f"z = x > 0: \n{z}\n")
z = x < 0  # Returns [False, False, False]
print(f"z = x < 0: \n{z}\n")

# -- Matrix Multiplication --
x1 = torch.rand((2, 5))
print(f"x1 = torch.rand((2, 5)): \n{x1}\n")
x2 = torch.rand((5, 3))
print(f"x2 = torch.rand((5, 3)): \n{x2}\n")
x3 = torch.mm(x1, x2)  # Matrix multiplication of x1 and x2, out shape: 2x3
print(f"x3 = torch.mm(x1, x2): \n{x3}\n")
x3 = x1.mm(x2)  # Similar as line above
print(f"x3 = x1.mm(x2): \n{x3}\n")

# -- Matrix Exponentiation --
matrix_exp = torch.rand(5, 5)
print(f"matrix_exp = torch.rand(5, 5): \n{matrix_exp}\n")
matrix_exp2 = matrix_exp.matrix_power(3)  # is same as torch.mm(matrix_exp,matrix_exp).mm(matrix_exp)
print(f"matrix_exp2 = matrix_exp.matrix_power(3): \n{matrix_exp2}\n")

# -- Element wise Multiplication --
z = x * y  # z = [9, 16, 21] = [1*9, 2*8, 3*7]
print(f"z = x * y: \n{z}\n")

# -- Dot product --
z = torch.dot(x, y)  # Dot product, in this case z = 1*9 + 2*8 + 3*7
print(f"z = torch.dot(x, y): \n{z}\n")

# -- Batch Matrix Multiplication --
batch = 2
n = 5
m = 4
p = 3
tensor1 = torch.rand((batch, n, m))
print(f"tensor1 = torch.rand((batch, n, m)): \n{tensor1}\n")
tensor2 = torch.rand((batch, m, p))
print(f"tensor2 = torch.rand((batch, m, p)): \n{tensor2}\n")
out_bmm = torch.bmm(tensor1, tensor2)  # Will be shape: (b x n x p)
print(f"out_bmm = torch.bmm(tensor1, tensor2): \n{out_bmm}\n")

# -- Example of broadcasting --
x1 = torch.rand((5, 5))
print(f"x1 = torch.rand((5, 5)): \n{x1}\n")
x2 = torch.ones((1, 5))
print(f"x2 = torch.ones((1, 5)): \n{x2}\n")
z = (x1 - x2)  # Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
print(f"z = (x1 - x2): \n{z}\n")
z = (x1**x2)  # Shape of z is 5x5: How? Broadcasting! Element wise exponentiation for every row
print(f"z = (x1 ** x2): \n{z}\n")

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)  # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
print(f"sum_x = torch.sum(x, dim=0): \n{sum_x}\n")
values, indices = torch.max(x, dim=0)  # Can also do x.max(dim=0)
print(f"values, indices = torch.max(x, dim=0): \n{values}\n{indices}\n")
values, indices = torch.min(x, dim=0)  # Can also do x.min(dim=0)
print(f"values, indices = torch.min(x, dim=0): \n{values}\n{indices}\n")
abs_x = torch.abs(x)  # Returns x where abs function has been applied to every element
print(f"abs_x = torch.abs(x): \n{abs_x}\n")
z = torch.argmax(x, dim=0)  # Gets index of the maximum value
print(f"z = torch.argmax(x, dim=0): \n{z}\n")
z = torch.argmin(x, dim=0)  # Gets index of the minimum value
print(f"z = torch.argmin(x, dim=0): \n{z}\n")
mean_x = torch.mean(x.float(), dim=0)  # mean requires x to be float
print(f"mean_x = torch.mean(x.float(), dim=0): \n{mean_x}\n")
z = torch.eq(x, y)  # Element wise comparison, in this case z = [False, False, False]
print(f"z = torch.eq(x, y): \n{z}\n")
sorted_y, indices = torch.sort(y, dim=0, descending=False)
print(f"sorted_y, indices = torch.sort(y, dim=0, descending=False): \n{sorted_y}\n{indices}\n")

z = torch.clamp(x, min=0)
print(f"z = torch.clamp(x, min=0): \n{z}\n")
# All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
# If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
print(f"x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool): \n{x}\n")
z = torch.any(x)  # will return True, can also do x.any() instead of torch.any(x)
print(f"z = torch.any(x): \n{z}\n")
z = torch.all(x)  # will return False (since not all are True), can also do x.all() instead of torch.all()
print(f"z = torch.all(x): \n{z}\n")

# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #

batch_size = 3
features = 10
print(f"batch_size = {batch_size}, features = {features}")
x = torch.rand((batch_size, features))
print(f"x = torch.rand((batch_size, features)): \n{x}\n")

# Get first examples features
print(f"x[0].shape: {x[0].shape}")  # shape [25], this is same as doing x[0,:]

# Get the first feature for all examples
print(f"x[:, 0].shape: {x[:, 0].shape}")  # shape [10]

# For example: Want to access third example in the batch and the first ten features
print(f"x[2, 0:10].shape: {x[2, 0:10].shape}")  # shape: [10]

# For example we can use this to, assign certain elements
x[0, 0] = 100
print(f"x[0, 0] = 100: \n{x}\n")  # shape: [10]

# Fancy Indexing
x = torch.arange(10)
print(f"x = torch.arange(10): \n{x}\n")
indices = [2, 5, 8]
print(f"indices = [2, 5, 8]: \n{indices}\n")

x = torch.rand((3, 5))
print(f"x = torch.rand((3, 5)): \n{x}\n")
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(f"rows = torch.tensor([1, 0]): \n{rows}\n")
print(f"cols = torch.tensor([4, 0]): \n{cols}\n")
print(f"x[rows, cols]: \n{x[rows, cols]}\n")  # Gets second row fifth column and first row first column

# More advanced indexing
x = torch.arange(10)
print(f"x = torch.arange(10): \n{x}\n")
print(f"x[(x < 2) | (x > 8)]: \n{x[(x < 2) | (x > 8)]}\n")  # will be [0, 1, 9]
print(f"x[x.remainder(2) == 0]: \n{x[x.remainder(2) == 0]}\n")

# Useful operations for indexing
print(f"torch.where(x > 5, x, x * 2): \n{torch.where(x > 5, x, x * 2)}\n")  # gives [0, 2, 4, 6, 8, 10, 6, 7, 8, 9], all values x > 5 yield x, else x*2
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]
print(f"x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique(): \n{x}\n")
print(f"x.ndimension(): \n{x.ndimension()}\n")
x = torch.arange(10)
print(f"x = torch.arange(10): \n{x}\n")
print(f"x.numel(): \n{x.numel()}\n") # The number of elements in x (in this case it's trivial because it's just a vector)

# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #

x = torch.arange(9)
print(f"x = torch.arange(9): \n{x}\n")

# Let's say we want to reshape it to be 3x3
x_3x3 = x.view(3, 3)
print(f"x_3x3 = x.view(3, 3): \n{x_3x3}\n")

# We can also do (view and reshape are very similar)
# and the differences are in simple terms (I'm no expert at this),
# is that view acts on contiguous tensors meaning if the
# tensor is stored contiguously in memory or not, whereas
# for reshape it doesn't matter because it will copy the
# tensor to make it contiguously stored, which might come
# with some performance loss.
# https://blog.csdn.net/qq_42761751/article/details/144194736
# https://blog.csdn.net/qq_28949847/article/details/128568811
# 功能同reshape相似，但是 view()只能操作 tensor，reshape()可以操作 tensor 和 ndarray。view()只能用在 contiguous (连续)的变量上。如果在 view 之前用了 transpose,permute 等切片处理，需要用 contiquous()来返回-个 contiguous copy。pytorch园 中的 torch.reshape()大致相当于 tensor.contiguous().view()
x_3x3 = x.reshape(3, 3)
print(f"x_3x3 = x.reshape(3, 3): \n{x_3x3}\n")

# If we for example do:
y = x_3x3.t()
print(f"y = x_3x3.t(): \n{y}\n")
print(f"y.is_contiguous(): \n{y.is_contiguous()}\n")  # This will return False and if we try to use view now, it won't work!
# y.view(9) would cause an error, reshape however won't

# This is because in memory it was stored [0, 1, 2, ... 8], whereas now it's [0, 3, 6, 1, 4, 7, 2, 5, 8]
# The jump is no longer 1 in memory for one element jump (matrices are stored as a contiguous block, and
# using pointers to construct these matrices). This is a bit complicated and I need to explore this more
# as well, at least you know it's a problem to be cautious of! A solution is to do the following
print(f"y.contiguous().view(9): \n{y.contiguous().view(9)}\n")  # Calling .contiguous() before view and it works

# Moving on to another operation, let's say we want to add two tensors dimensions togethor
x1 = torch.rand(2, 5)
print(f"x1 = torch.rand(2, 5): \n{x1}\n")
x2 = torch.rand(2, 5)
print(f"x2 = torch.rand(2, 5): \n{x2}\n")
print(f"torch.cat((x1, x2), dim=0).shape: {torch.cat((x1, x2), dim=0).shape}")  # Shape: 4x5
print(f"torch.cat((x1, x2), dim=1).shape: {torch.cat((x1, x2), dim=1).shape}")  # Shape 2x10

# Let's say we want to unroll x1 into one long vector with 10 elements, we can do:
z = x1.view(-1)  # And -1 will unroll everything
print(f"z = x1.view(-1): \n{z}\n")

# If we instead have an additional dimension and we wish to keep those as is we can do:
batch = 2
x = torch.rand((batch, 4, 5))
print(f"x = torch.rand((batch, 4, 5)): \n{x}\n")
z = x.view(batch, -1)  # And z.shape would be 2x20, this is very useful stuff and is used all the time
print(f"z = x.view(batch, -1): \n{z}\n")

# Let's say we want to switch x axis so that instead of 2x4x5 we have 2x5x4
# I.e we want dimension 0 to stay, dimension 1 to become dimension 2, dimension 2 to become dimension 1
# Basically you tell permute where you want the new dimensions to be, torch.transpose is a special case
# of permute (why?)
z = x.permute(0, 2, 1)
print(f"z = x.permute(0, 2, 1): \n{z}\n")

# Splits x last dimension into chunks of 2 (since 5 is not integer div by 2) the last dimension
# will be smaller, so it will split it into two tensors: 2x2x5 and 2x2x5
z = torch.chunk(x, chunks=2, dim=1)
print(f"z = torch.chunk(x, chunks=2, dim=1): \n{z}\n")
print(f"z[0].shape: \n{z[0].shape}\n")
print(f"z[1].shape: \n{z[1].shape}\n")

# Let's say we want to add an additional dimension
x = torch.arange(10)  # Shape is [10], let's say we want to add an additional so we have 1x10
print(f"x = torch.arange(10): \n{x}\n")
print(f"x.unsqueeze(0).shape: \n{x.unsqueeze(0).shape}\n")
print(f"x.unsqueeze(1).shape: \n{x.unsqueeze(1).shape}\n") # 10x1

# Let's say we have x which is 1x1x10 and we want to remove a dim so we have 1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(f"x = torch.arange(10).unsqueeze(0).unsqueeze(1): \n{x}\n")
print(f"x.shape: \n{x.shape}\n")

# Perhaps unsurprisingly
z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10
print(f"z = x.squeeze(1): \n{z}\n")
print(f"z.shape: \n{z.shape}\n")

# That was some essential Tensor operations, hopefully you found it useful!
