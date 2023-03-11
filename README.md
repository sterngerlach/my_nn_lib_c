
# my-nn-lib-c

A naive and stupid implementation of neural networks using C

## Build from source

Build the library and examples using the following commands:
```Bash
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j$(nproc)
```

## Run examples

Assume that the MNIST dataset is placed in `/path/to/mnist/`:
```Bash
$ ls -1 /path/to/mnist
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
train-images-idx3-ubyte
train-labels-idx1-ubyte
```

Train the simple neural network using the following command:
```Bash
$ ./build/examples/train_dnn_mnist /path/to/mnist/
```

Train the LeNet-5 model using the following command:
```Bash
$ ./build/examples/train_lenet5_mnist /path/to/mnist/
```
