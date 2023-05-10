#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "params/0_bias_lut.h"
#include "params/0_bias_indices.h"
#include "params/3_bias_lut.h"
#include "params/3_bias_indices.h"
#include "params/6_bias_lut.h"
#include "params/6_bias_indices.h"
#include "params/0_weight_lut.h"
#include "params/0_weight_indices.h"
#include "params/3_weight_lut.h"
#include "params/3_weight_indices.h"
#include "params/6_weight_lut.h"
#include "params/6_weight_indices.h"
#include "params/classifier_1_weight_lut.h"
#include "params/classifier_1_weight_indices.h"
#include "params/classifier_2_weight_lut.h"
#include "params/classifier_2_weight_indices.h"
#include "params/classifier_1_bias_lut.h"
#include "params/classifier_1_bias_indices.h"
#include "params/classifier_2_bias_lut.h"
#include "params/classifier_2_bias_indices.h"


#include "params/0_bias.h"
#include "params/3_bias.h"
#include "params/6_bias.h"
#include "params/0_weight.h"
#include "params/3_weight.h"
#include "params/6_weight.h"
#include "params/classifier_1_weight.h"
#include "params/classifier_2_weight.h"
#include "params/classifier_1_bias.h"
#include "params/classifier_2_bias.h"

#include "params/dataset120_eval.h" // 3923 
#define DATASET_UNITS 3923


#define INPUT_SIZE 40 //120
#define NUM_FILTERS 64
#define KERNEL_SIZE 5
#define CONV0_INPUT_SIZE 40  // 120 
#define CONV3_INPUT_SIZE 36 // 116
#define CONV6_INPUT_SIZE 32 // 112 
#define FC1_OUTPUT_SIZE  128
#define FC2_OUTPUT_SIZE 5


#define QUANTIZATION 1  /// 1 = ON    0 = OFF



void print_confusion_matrix(int *expected, int *predicted, int n) {
    int matrix[n][n];
    int i, j;
    
    // Inicializa a matriz de confusão com zeros
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i][j] = 0;
        }
    }
    
    // Preenche a matriz de confusão com os valores esperados e calculados
    for (i = 0; i < DATASET_UNITS; i++) {
        matrix[expected[i]][predicted[i]]++;
    }
    
    // Imprime a matriz de confusão
    printf("Confusion Matrix:\n");
    printf("   ");
    for (i = 0; i < n; i++) {
        printf("   %d ", i);
    }
    printf("\n\n");
    for (i = 0; i < n; i++) {
        printf("%d: ", i);
        for (j = 0; j < n; j++) {
            printf("   %d ", matrix[i][j]);
        }
        printf("\n");
    }
}



main(){
    printf("Boot... ");
    int i = 0;
    float input_vector[INPUT_SIZE]; 
    int targetLabel;
    int correctLabels = 0;
    int wrongLabels = 0;
    int predictedList[DATASET_UNITS];
    int expectedList[DATASET_UNITS];
    int listIndex = 0;
    printf("Evaluating...\n");

for(int datasetIndex = 0 ; datasetIndex < DATASET_UNITS ; datasetIndex++ ){

    float progress = datasetIndex / (float)DATASET_UNITS;
    printf("Progress %0.f\n",progress*100);
    progress = (datasetIndex/DATASET_UNITS)*100;


    int startingIndex = datasetIndex * (INPUT_SIZE + 1); //input + label
    targetLabel = dataset120[startingIndex + INPUT_SIZE];

    for(i = 0 ; i < INPUT_SIZE ; i++){
        input_vector[i] = dataset120[startingIndex + i];
    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 1
// CONVOLUTION 1
    float conv0_currentKernel[KERNEL_SIZE];
    float conv0_current_bias = 0;
    float conv0_featureMap[NUM_FILTERS][CONV0_INPUT_SIZE-4];

    for (int k = 0; k < NUM_FILTERS; k++)
    {
        // Load Current Weights
        for (i = 0; i < KERNEL_SIZE; i++)
        {
            if ( QUANTIZATION == 1 ){
                conv0_currentKernel[i] =   conv0_weights_lut[conv0_weights_indices[i + (k * KERNEL_SIZE)]];
            } else {
                conv0_currentKernel[i] =   conv0_weights[i + (k * KERNEL_SIZE)];
            }
            // printf("%f   %f (%d)  \n",conv0_weights[i + (k * KERNEL_SIZE)], conv0_weights_lut[ conv0_weights_indices[i + (k * KERNEL_SIZE)] ],conv0_weights_indices[i + (k * KERNEL_SIZE)] );
        }

        // Load Current Bias
        if ( QUANTIZATION == 1 ){
            conv0_current_bias = conv0_bias_lut[conv0_bias_indices[k]];
        } else {
            conv0_current_bias = conv0_bias[k];
        }
        // printf("%f   %f (%d)    Err:%f \n",conv0_bias[k], conv0_bias_lut[conv0_bias_indices[k]],conv0_bias_indices[k], conv0_bias[k]-conv0_bias_lut[conv0_bias_indices[k]] );
        


        // Perform Kernel operation
        for (i = 0; i <= sizeof(input_vector)/sizeof(input_vector[0])-KERNEL_SIZE; i++)
        {
            float totalSum = 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
               totalSum += input_vector[i+j] * conv0_currentKernel[j];
            }

            conv0_featureMap[k][i] = totalSum + conv0_current_bias;
        }
    }
///// RELU

///// RELU
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        for (int j = 0; j < CONV0_INPUT_SIZE-4; j++)
        {
            
            if (conv0_featureMap[i][j] <= 0)
                conv0_featureMap[i][j] = 0;    
        }
    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 3
// CONVOLUTION 3
    float conv3_featureMap[NUM_FILTERS][CONV3_INPUT_SIZE-4]; // 112



    float conv3_totalSum = 0;
    for (int filterToGenerate = 0 ; filterToGenerate < NUM_FILTERS ; filterToGenerate ++ ){
        for (int inputOffset = 0 ; inputOffset < CONV3_INPUT_SIZE-4 ; inputOffset++){
            conv3_totalSum = 0;
            for (int filterIn = 0 ; filterIn < NUM_FILTERS ; filterIn++){
                for (int kernelIndex = 0 ; kernelIndex < KERNEL_SIZE ; kernelIndex++){
                    int weightIndex = kernelIndex + (filterIn * KERNEL_SIZE) + ( filterToGenerate * NUM_FILTERS * KERNEL_SIZE ) ;
                    int indexIn = kernelIndex + (inputOffset);

                    if ( QUANTIZATION == 1) {
                         conv3_totalSum += conv0_featureMap[filterIn][indexIn] * conv3_weights_lut[conv3_weights_indices[weightIndex]];
                    } else {
                        conv3_totalSum += conv0_featureMap[filterIn][indexIn] * conv3_weights[weightIndex]; 
                    }
                    // printf("%f   %f (%d)    Err:%f \n", conv3_weights[weightIndex] , conv3_weights_lut[conv3_weights_indices[weightIndex]], conv3_weights_indices[weightIndex], conv3_weights[weightIndex] - conv3_weights_lut[conv3_weights_indices[weightIndex]] );
                    // printf("%f   %f (%d)    Err:%f \n", aaa , bbbb, ccccc, aaa - bbb );
                }
            }

            if ( QUANTIZATION == 1) {
                conv3_totalSum += conv3_bias_lut[conv3_bias_indices[filterToGenerate]];
            } else {
                conv3_totalSum += conv3_bias[filterToGenerate];
            }            
            // printf("%f   %f (%d)    Err:%f \n", conv3_bias[filterToGenerate] , conv3_bias_lut[conv3_bias_indices[filterToGenerate]], conv3_bias_indices[filterToGenerate], conv3_bias[filterToGenerate] - conv3_bias_lut[conv3_bias_indices[filterToGenerate]] );
            conv3_featureMap[filterToGenerate][inputOffset] = conv3_totalSum;
        }
    }
    
    

    
///// RELU
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        for (int j = 0; j < CONV3_INPUT_SIZE-4; j++)
        {
            if (conv3_featureMap[i][j] <= 0)
                conv3_featureMap[i][j] = 0;        
        }
    }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 6
// CONVOLUTION 6

    float conv6_featureMap[NUM_FILTERS][CONV6_INPUT_SIZE-4]; 


    float totalSum = 0;
    for (int filterToGenerate = 0 ; filterToGenerate < NUM_FILTERS ; filterToGenerate ++ ){
        for (int inputOffset = 0 ; inputOffset < CONV6_INPUT_SIZE-4 ; inputOffset++){
            totalSum = 0;
            for (int filterIn = 0 ; filterIn < NUM_FILTERS ; filterIn++){
                for (int kernelIndex = 0 ; kernelIndex < KERNEL_SIZE ; kernelIndex++){
                    int weightIndex = kernelIndex + (filterIn * KERNEL_SIZE) + ( filterToGenerate * NUM_FILTERS * KERNEL_SIZE ) ;
                    int indexIn = kernelIndex + (inputOffset);

                    if ( QUANTIZATION == 1) {
                        totalSum += conv3_featureMap[filterIn][indexIn] * conv6_weights_lut[conv6_weights_indices[weightIndex]]; 
                    } else {
                        totalSum += conv3_featureMap[filterIn][indexIn] * conv6_weights[weightIndex]; 
                    }                   
                    // printf("%f   %f (%d)    Err:%f \n", conv6_weights[weightIndex] , conv6_weights_lut[conv6_weights_indices[weightIndex]], conv6_weights_indices[weightIndex], conv6_weights[weightIndex] - conv6_weights_lut[conv6_weights_indices[weightIndex]] );
                }
            }
            if ( QUANTIZATION == 1) {
                totalSum += conv6_bias_lut[conv6_bias_indices[filterToGenerate]];
            } else {
                totalSum += conv6_bias[filterToGenerate];
            }     
            // printf("%f   %f (%d)    Err:%f \n", conv6_bias[filterToGenerate] , conv6_bias_lut[conv6_bias_indices[filterToGenerate]], conv6_bias_indices[filterToGenerate], conv6_bias[filterToGenerate] - conv6_bias_lut[conv6_bias_indices[filterToGenerate]] );
            // printf("%f   %f (%d)    Err:%f \n", aaa , bbbb, ccccc, aaa - bbb );
            conv6_featureMap[filterToGenerate][inputOffset] = totalSum;
        }
    }
    
    
/////////////////////////// RELU
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        for (int j = 0; j < CONV6_INPUT_SIZE-4; j++)
        {
            if (conv6_featureMap[i][j] <= 0)
                conv6_featureMap[i][j] = 0;            
        }
    }



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 2
////////////////////////////////////////// FC Layer
///// Flatten

    float flatten1_vector[NUM_FILTERS*(CONV6_INPUT_SIZE-4)];
    int index = 0;

    for (int k = 0; k < NUM_FILTERS; k++)
    {
        for (int i = 0; i < CONV6_INPUT_SIZE-4; i++)
        {
            flatten1_vector[index] =  conv6_featureMap[k][i];
            index++;
        }
    }
    
/////// FC 1 (6192 > 128)

    float fc1_out_vector[FC1_OUTPUT_SIZE];
    float totalValue;
    int fc1_inputSize = NUM_FILTERS*(CONV6_INPUT_SIZE-4); // 6912


    for (int outputIndex = 0; outputIndex < FC1_OUTPUT_SIZE; outputIndex++){
        totalValue = 0;
        for (int i = 0; i < fc1_inputSize; i++)
        {
            if(QUANTIZATION == 1){
                totalValue += flatten1_vector[i] * fc1_weights_lut[fc1_weights_indices[(fc1_inputSize*outputIndex)+i]];
            }else{
                totalValue += flatten1_vector[i] * fc1_weights[(fc1_inputSize*outputIndex)+i];
            }
            //printf("%f   %f (%d)    Err:%f \n", fc1_weights[(fc1_inputSize*outputIndex)+i] , fc1_weights_lut[fc1_weights_indices[(fc1_inputSize*outputIndex)+i]], fc1_weights_indices[(fc1_inputSize*outputIndex)+i], fc1_weights[(fc1_inputSize*outputIndex)+i] - fc1_weights_lut[fc1_weights_indices[(fc1_inputSize*outputIndex)+i]] );
            // printf("%f   %f (%d)    Err:%f \n", aaa , bbbb, ccccc, aaa - bbb );
        }

        if(QUANTIZATION == 1){
            fc1_out_vector[outputIndex] = totalValue + fc1_bias_lut[fc1_bias_indices[outputIndex]];
        }else{
            fc1_out_vector[outputIndex] = totalValue + fc1_bias[outputIndex];
        }
    }


    
////////////////// RELU
    for (int i = 0; i < FC1_OUTPUT_SIZE ; i++){
        if ( fc1_out_vector[i] < 0) fc1_out_vector[i] = 0;        
    }



///////////////// FC 1 (128 > 5)
    float fc2_out_vector[FC2_OUTPUT_SIZE];
    float fc2_totalValue;
     
    for (int outputIndex = 0; outputIndex < FC2_OUTPUT_SIZE; outputIndex++){
        fc2_totalValue = 0;
        for (int i = 0; i < FC1_OUTPUT_SIZE; i++)
        {
            if(QUANTIZATION == 1){
                fc2_totalValue += fc1_out_vector[i] * fc2_weights_lut[fc2_weights_indices[(FC1_OUTPUT_SIZE*outputIndex)+i]];
            }else{
                fc2_totalValue += fc1_out_vector[i] * fc2_weights[(FC1_OUTPUT_SIZE*outputIndex)+i];
            }
        }
        if(QUANTIZATION == 1){
            fc2_out_vector[outputIndex] = fc2_totalValue + fc2_bias_lut[fc2_bias_indices[outputIndex]];
        }else{
        fc2_out_vector[outputIndex] = fc2_totalValue + fc2_bias[outputIndex];
        }
    }

    

////////////////////////// Result Classes
    //printf("\n############################################################## RESULT CLASSES");


    float maxValue = fc2_out_vector[0];
    int calculatedLabel = 0;
 

    for (int i = 0; i < 5; i++){
        if (fc2_out_vector[i] > maxValue){
             maxValue = fc2_out_vector[i];
             calculatedLabel = i;
        }
    }

    if ( targetLabel == calculatedLabel){
        printf("Correct prediction (predicted %d) (correct %d)\n",calculatedLabel,targetLabel);
        correctLabels++;
    } else {
        printf("Wrong prediction  (predicted %d) (correct %d)\n",calculatedLabel,targetLabel);
        wrongLabels++;
    }

    predictedList[listIndex] = calculatedLabel;
    expectedList[listIndex] = targetLabel;
    listIndex++;


}
    printf("----------------------\n");
    printf("Accuracy: %0.1f\n",(float)correctLabels/(float)DATASET_UNITS*100);
    printf("----------\n");
    printf("Correct predictions: %d \n",correctLabels);
    printf("Wrong predictions: %d \n",wrongLabels);
    printf("----------------------\n");
    printf("Confusion Matrix\n");
    print_confusion_matrix(&predictedList,&expectedList,5);



    printf(" \n------------------------------------------------------------- End..");
    return 0;
}



