# Quantized CNN integer
This branch aims to generate an quantized model using LUT tables in integer form

### To run this branch:
First, set CALCULATE_FLOAT in model.c accordingly


##### Example 64 bins with variable size bins
> cd params; ./quantizate 64 variable; cd ..; ./compile; ./model;

##### Example 128 bins with fixed size bins
> cd params; ./quantizate 128 fixed; cd ..; ./compile; ./model;

###### Note: The model will run even if ./compile fails (with previous working version), if changes are being made in the model.c file make sure compilation works before running the above lines
###### Note: Compile bash contains a supress all warnings command to prevent long compile times due the weights in #include BE CAREFUL


## Aditional Scripts

### convertTensor.py
This script converts the pytorch extracted tensor and generates an array file to be included in C
###### Note: If error is found due to wrong character detected, check if the end of the tensor contains an cuda configuration, if YES, remove it. (Remove the object, including the ",")

### quantizate.py 
This scripts converts .h weight/bias files generated using convertTensor into two files indices and lut files, number of bins and quantization mode is chosen on launch
###### Note: The script DO NOT support scientific notation numbers (e.g. -1.5105e01) ALL values must be without any kind of notation, just numbers.
