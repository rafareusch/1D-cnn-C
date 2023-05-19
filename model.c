#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#define BATCH_SIZE 32 
#define INPUT_SIZE 120
#define NUM_FILTERS 64
#define KERNEL_SIZE 5
#define CONV0_INPUT_SIZE 120 
#define CONV3_INPUT_SIZE 116
#define CONV6_INPUT_SIZE 112 
#define FC1_OUTPUT_SIZE 128
#define FC2_OUTPUT_SIZE 5





main(){
    int i = 0;
    float input_vector[INPUT_SIZE]; 
    int targetLabel;

    for(int datasetIndex = 0 ; datasetIndex < DATASET_UNITS ; datasetIndex++ ){

        int startingIndex = datasetIndex * 121; //input + label
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
                conv0_currentKernel[i] = conv0_weights[i + (k * KERNEL_SIZE)];
            }

            // Load Current Bias
            conv0_current_bias = conv0_bias[k];
            


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
                        conv3_totalSum += conv0_featureMap[filterIn][indexIn] * conv3_weights[weightIndex]; 

                    }
                }
                conv3_totalSum += conv3_bias[filterToGenerate];
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
                        totalSum += conv3_featureMap[filterIn][indexIn] * conv6_weights[weightIndex]; 
                    }
                }
                totalSum += conv6_bias[filterToGenerate];
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
                totalValue += flatten1_vector[i] * fc1_weights[(fc1_inputSize*outputIndex)+i];
            }
            fc1_out_vector[outputIndex] = totalValue + fc1_bias[outputIndex];
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
                fc2_totalValue += fc1_out_vector[i] * fc2_weights[(FC1_OUTPUT_SIZE*outputIndex)+i];
            }
            fc2_out_vector[outputIndex] = fc2_totalValue + fc2_bias[outputIndex];
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
        printf("%d,%d,",calculatedLabel,targetLabel);
     
    }

    return 0;
}



