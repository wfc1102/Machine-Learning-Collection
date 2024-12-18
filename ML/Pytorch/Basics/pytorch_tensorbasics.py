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

# Initializing a Tensor in this case of shape 2x3 (2 rows, 3 columns)
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# A few tensor attributes
print(
    f"Information about tensor: {my_tensor}"
)  # Prints data of the tensor, device and grad info
print(
    f"Type of Tensor {my_tensor.dtype}"
)  # Prints dtype of the tensor (torch.float32, etc)
print(
    f"Device Tensor is on {my_tensor.device}"
)  # Prints cpu/cuda (followed by gpu number)
print(f"Shape of tensor {my_tensor.shape}")  # Prints shape, in this case 2x3
print(f"Requires gradient: {my_tensor.requires_grad}")  # Prints true/false

# Other common initialization methods (there exists a ton more)
x = torch.empty(size=(3, 3))  # Tensor of shape 3x3 with uninitialized data
print(f"torch.empty:")
print(x)
x = torch.zeros((3, 3))  # Tensor of shape 3x3 with values of 0
print(f"torch.zeros: ")
print(x)
x = torch.rand(
    (3, 3)
)  # Tensor of shape 3x3 with values from uniform distribution in interval [0,1)
print(f"torch.rand: ")
print(x)
x = torch.ones((3, 3))  # Tensor of shape 3x3 with values of 1
print(f"torch.ones: ")
print(x)
x = torch.eye(5, 5)  # Returns Identity Matrix I, (I <-> Eye), matrix of shape 2x3
print(f"torch.eye: ")
print(x)
x = torch.arange(
    start=0, end=5, step=1
)  # Tensor [0, 1, 2, 3, 4], note, can also do: torch.arange(11)
print(f"torch.arange: ")
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
print(f"torch.linspace: ")
print(x)
x = torch.empty(size=(1, 5)).normal_(
    mean=0, std=1
)  # Normally distributed with mean=0, std=1
print(f"torch.empty.normal_: ")
print(x)
x = torch.empty(size=(1, 5)).uniform_(
    0, 1
)  # Values from a uniform distribution low=0, high=1
print(f"torch.empty.uniform_: ")
print(x)
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3
print(f"torch.diag: ")
print(x)

# How to make initialized tensors to other types (int, float, double)
# These will work even if you're on CPU or CUDA!
tensor = torch.arange(4)  # [0, 1, 2, 3] Initialized as int64 by default
print(f"Converted Boolean: {tensor.bool()}")  # Converted to Boolean: 1 if nonzero
print(f"Converted int16 {tensor.short()}")  # Converted to int16
print(
    f"Converted int64 {tensor.long()}"
)  # Converted to int64 (This one is very important, used super often)
print(f"Converted float16 {tensor.half()}")  # Converted to float16
print(
    f"Converted float32 {tensor.float()}"
)  # Converted to float32 (This one is very important, used super often)
print(f"Converted float64 {tensor.double()}")  # Converted to float64

# Array to Tensor conversion and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_again = (
    tensor.numpy()
)  # np_array_again will be same as np_array (perhaps with numerical round offs)

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
print(x)
y = torch.tensor([9, 8, 7])

# -- Addition --
z1 = torch.empty(3)
torch.add(x, y, out=z1)  # This is one way
z2 = torch.add(x, y)  # This is another way
z = x + y  # This is my preferred way, simple and clean.

# -- Subtraction --
z = x - y  # We can do similarly as the preferred way of addition

# -- Division (A bit clunky) --
z = torch.true_divide(x, y)  # Will do element wise division if of equal shape

# -- Inplace Operations --
t = torch.zeros(3)
print(t)

t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
print("t.add_(x)")
print(t)
t += x  # Also inplace: t = t + x is not inplace, bit confusing.
print("t += x")
print(t)
#t = t + x
#print("t = t + x")
#print(t)

# -- Exponentiation (Element wise if vector or matrices) --
z = x.pow(2)  # z = [1, 4, 9]
print("x.pow(2)")
print(z)
z = x**2  # z = [1, 4, 9]
print("x**2")
print(z)


# -- Simple Comparison --
z = x > 0  # Returns [True, True, True]
print("x>0")
print(z)
z = x < 0  # Returns [False, False, False]
print("x<0")
print(z)

# -- Matrix Multiplication --
x1 = torch.rand((2, 5))
print("torch.rand((2, 5))")
print(x1)
x2 = torch.rand((5, 3))
print("torch.rand((5, 3))")
print(x2)
x3 = torch.mm(x1, x2)  # Matrix multiplication of x1 and x2, out shape: 2x3
print("torch.mm(x1, x2)")
print(x3)
x3 = x1.mm(x2)  # Similar as line above
print("x1.mm(x2)")
print(x3)

# -- Matrix Exponentiation --
matrix_exp = torch.rand(5, 5)
print(
    matrix_exp.matrix_power(3)
)  # is same as matrix_exp (mm) matrix_exp (mm) matrix_exp

# -- Element wise Multiplication --
z = x * y  # z = [9, 16, 21] = [1*9, 2*8, 3*7]
print("x*y")
print(z)

# -- Dot product --
z = torch.dot(x, y)  # Dot product, in this case z = 1*9 + 2*8 + 3*7
print("torch.dot(x, y)")
print(z)

# -- Batch Matrix Multiplication --
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
print("torch.rand((batch, n, m))")
print(tensor1)
tensor2 = torch.rand((batch, m, p))
print("torch.rand((batch, m, p))")
print(tensor2)
out_bmm = torch.bmm(tensor1, tensor2)  # Will be shape: (b x n x p)
print("torch.bmm(tensor1, tensor2)")
print(out_bmm)

# -- Example of broadcasting --
x1 = torch.rand((5, 5))
print("torch.rand((5, 5))")
print(x1)
x2 = torch.ones((1, 5))
print("torch.ones((1, 5))")
print(x2)
z = (
    x1 - x2
)  # Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
print("x1 - x2")
print(z)
z = (
    x1**x2
)  # Shape of z is 5x5: How? Broadcasting! Element wise exponentiation for every row
print("x1 ** x2")
print(z)

# Other useful tensor operations
sum_x = torch.sum(
    x, dim=0
)  # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
print("torch.sum(x, dim=0)")
print(sum_x)
values, indices = torch.max(x, dim=0)  # Can also do x.max(dim=0)
print("torch.max(x, dim=0)")
print(values)
print(indices)
values, indices = torch.min(x, dim=0)  # Can also do x.min(dim=0)
print("torch.min(x, dim=0)")
print(values)
print(indices)
abs_x = torch.abs(x)  # Returns x where abs function has been applied to every element
print("torch.abs(x)")
print(abs_x)
z = torch.argmax(x, dim=0)  # Gets index of the maximum value
print("torch.argmax(x, dim=0)")
print(z)
z = torch.argmin(x, dim=0)  # Gets index of the minimum value
print("torch.argmin(x, dim=0)")
print(z)
mean_x = torch.mean(x.float(), dim=0)  # mean requires x to be float
print("torch.mean(x.float(), dim=0)")
print(mean_x)
z = torch.eq(x, y)  # Element wise comparison, in this case z = [False, False, False]
print("torch.eq(x, y)")
print(z)
sorted_y, indices = torch.sort(y, dim=0, descending=False)
print("torch.sort(y, dim=0, descending=False)")
print(sorted_y)
print(indices)

z = torch.clamp(x, min=0)
print("torch.clamp(x, min=0)")
print(z)
# All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
# If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
print("torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)")
print(x)
z = torch.any(x)  # will return True, can also do x.any() instead of torch.any(x)
print("torch.any(x)")
print(z)
z = torch.all(
    x
)  # will return False (since not all are True), can also do x.all() instead of torch.all()
print("torch.all(x)")
print(z)

# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print("torch.rand((batch_size, features))")
print(x)

# Get first examples features
print(x[0].shape)  # shape [25], this is same as doing x[0,:]

# Get the first feature for all examples
print(x[:, 0].shape)  # shape [10]

# For example: Want to access third example in the batch and the first ten features
print(x[2, 0:10].shape)  # shape: [10]

# For example we can use this to, assign certain elements
x[0, 0] = 100

# Fancy Indexing
x = torch.arange(10)
print("torch.arange(10)")
print(x)
indices = [2, 5, 8]
print("x[indices]")
print(x[indices])  # x[indices] = [2, 5, 8]

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print("x rows cols")
print(x)
print(rows)
print(cols)
print("x[rows, cols]")
print(x[rows, cols])  # Gets second row fifth column and first row first column

# More advanced indexing
x = torch.arange(10)
print("torch.arange(10)")
print(x)
print("x[(x < 2) | (x > 8)]")
print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9]
print("x[x.remainder(2) == 0]")
print(x[x.remainder(2) == 0])  # will be [0, 2, 4, 6, 8]

# Useful operations for indexing
print("torch.where(x > 5, x, x * 2)")
print(
    torch.where(x > 5, x, x * 2)
)  # gives [0, 2, 4, 6, 8, 10, 6, 7, 8, 9], all values x > 5 yield x, else x*2
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]
print("torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()")
print(x)
print("x.ndimension()")
print(
    x.ndimension()
)  # The number of dimensions, in this case 1. if x.shape is 5x5x5 ndim would be 3
x = torch.arange(10)
print("torch.arange(10)")
print(x)
print("x.numel()")
print(
    x.numel()
)  # The number of elements in x (in this case it's trivial because it's just a vector)

# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #

x = torch.arange(9)
print("torch.arange(9)")
print(x)

# Let's say we want to reshape it to be 3x3
x_3x3 = x.view(3, 3)
print("x.view(3, 3)")
print(x_3x3)

# We can also do (view and reshape are very similar)
# and the differences are in simple terms (I'm no expert at this),
# is that view acts on contiguous tensors meaning if the
# tensor is stored contiguously in memory or not, whereas
# for reshape it doesn't matter because it will copy the
# tensor to make it contiguously stored, which might come
# with some performance loss.
x_3x3 = x.reshape(3, 3)
print("x.reshape(3, 3)")
print(x_3x3)

# If we for example do:
y = x_3x3.t()
print("x_3x3.t()")
print(y)
print("y.is_contiguous()")
print(
    y.is_contiguous()
)  # This will return False and if we try to use view now, it won't work!
# y.view(9) would cause an error, reshape however won't

# This is because in memory it was stored [0, 1, 2, ... 8], whereas now it's [0, 3, 6, 1, 4, 7, 2, 5, 8]
# The jump is no longer 1 in memory for one element jump (matrices are stored as a contiguous block, and
# using pointers to construct these matrices). This is a bit complicated and I need to explore this more
# as well, at least you know it's a problem to be cautious of! A solution is to do the following
print("y.contiguous().view(9)")
print(y.contiguous().view(9))  # Calling .contiguous() before view and it works

# Moving on to another operation, let's say we want to add two tensors dimensions togethor
x1 = torch.rand(2, 5)
print("y.contiguous().view(9)")
print(x1)
x2 = torch.rand(2, 5)
print("torch.rand(2, 5)")
print(x2)
print("torch.cat((x1, x2), dim=0).shape")
print(torch.cat((x1, x2), dim=0).shape)  # Shape: 4x5
print("torch.cat((x1, x2), dim=1).shape")
print(torch.cat((x1, x2), dim=1).shape)  # Shape 2x10

# Let's say we want to unroll x1 into one long vector with 10 elements, we can do:
z = x1.view(-1)  # And -1 will unroll everything
print("x1.view(-1)")
print(z)

# If we instead have an additional dimension and we wish to keep those as is we can do:
batch = 64
x = torch.rand((batch, 2, 5))
print("torch.rand((batch, 2, 5))")
print(x)
z = x.view(
    batch, -1
)  # And z.shape would be 64x10, this is very useful stuff and is used all the time
print("x.view(batch, -1)")
print(z)

# Let's say we want to switch x axis so that instead of 64x2x5 we have 64x5x2
# I.e we want dimension 0 to stay, dimension 1 to become dimension 2, dimension 2 to become dimension 1
# Basically you tell permute where you want the new dimensions to be, torch.transpose is a special case
# of permute (why?)
z = x.permute(0, 2, 1)
print("x.permute(0, 2, 1)")
print(z)

# Splits x last dimension into chunks of 2 (since 5 is not integer div by 2) the last dimension
# will be smaller, so it will split it into two tensors: 64x2x3 and 64x2x2
z = torch.chunk(x, chunks=2, dim=1)
print("torch.chunk(x, chunks=2, dim=1)")
print(z)
print("z[0].shape")
print(z[0].shape)
print("z[1].shape")
print(z[1].shape)

# Let's say we want to add an additional dimension
x = torch.arange(
    10
)  # Shape is [10], let's say we want to add an additional so we have 1x10
print("torch.arange(10)")
print(x)
print("x.unsqueeze(0).shape")
print(x.unsqueeze(0).shape)  # 1x10
print("x.unsqueeze(1).shape")
print(x.unsqueeze(1).shape)  # 10x1

# Let's say we have x which is 1x1x10 and we want to remove a dim so we have 1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print("torch.arange(10).unsqueeze(0).unsqueeze(1)")
print(x)

# Perhaps unsurprisingly
z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10
print("x.squeeze(1)")
print(z)

# That was some essential Tensor operations, hopefully you found it useful!
