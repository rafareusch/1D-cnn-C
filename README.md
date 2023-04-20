# 1D Convolutional Neural Network in C

This project aims to allow different the comparison of different post-training optimization techniques using C and it's the basis of my Master's Degree. These optimization are separated into branches.
This neural network model is custom designed using bare-metal C. It uses weights trained and extracted via Pytorch, these weights can be found in params folder.

## Branches

### main
This branch has the 'golden model', with operations happening the same as Pytorch inference.

### int-cnn
This branch consists of processing all the values with INTEGER variables, no float is used.
Float variables can be found in code, but they are only present for comparing how 'wrong' the values presented in integer really are.

### quantized-cnn
This branch aims to generate an quantized model using LUT tables and various techniques.


## Compile and Running Instructions
./compile
./model

##### Note: Compile bash contains a supress all warnings command to prevent long compile times due the weights #include



## Aditional Scripts

### convertTensor.py
This script converts the pytorch extracted tensor and generates an array file to be included in C
##### Note: If error is found due to wrong character detecter, check if the end of the tensor contains an cuda configuration, if YES, remove it. (Remove the object, including the ",")
