# CS512 Assignment 2 (CRFs and Convolutions)

## Implementation and Tests of Custom Convolution layer
- batch_custom_Conv.py : Implementation of Custom Conv2d for a given kernel_size,  zero-padding and stride
- conv_test.py: Required test case for Q3a.

## Implementation of CRF layer with Custom Conv Layer
- crf.py : Extends torch.nn.Module to define CRFLoss Function and custom crf layers
- crf_utils.py : Implementation of decoder (inference), loss calculation, gradient calculation, accuracy
- crf_tests.py : Test cases checking the functions in crf_utils as well as crf. Gradient check

## Implementation of CNN Architectures
- lenet.py : Architexture of LeNet (Implementation of LeNet)
- dCNN.py : Training and Plot

## training
- train.py : Modified starter code
