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
###### Some branches have different compiling instructions.
###### Note: Compile bash contains a supress all warnings command to prevent long compile times due the weights #include


## Aditional Scripts

### convertTensor.py
This script converts the pytorch extracted tensor and generates an array file to be included in C
###### Note: If error is found due to wrong character detected, check if the end of the tensor contains an cuda configuration, if YES, remove it. (Remove the object, including the ",")

### quantizate.py 
This scripts converts .h weight/bias files generated using convertTensor into two files indices and lut files, number of bins and quantization mode is chosen on launch
###### Note: The script DO NOT support scientific notation numbers (e.g. -1.5105e01) ALL values must be without any kind of notation, just numbers.
