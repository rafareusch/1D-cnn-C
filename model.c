#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "params/0_bias.h"
#include "params/0_weight.h"
#include "params/3_bias.h"
#include "params/3_weight.h"
#include "params/6_bias.h"
#include "params/6_weight.h"
#include "params/classifier_1_weight.h"
#include "params/classifier_1_bias.h"
#include "params/classifier_2_weight.h"
#include "params/classifier_2_bias.h"


#define INPUT_SIZE 120
#define NUM_FILTERS 64
#define KERNEL_SIZE 5
#define CONV0_INPUT_SIZE 120 
#define CONV3_INPUT_SIZE 116
#define CONV6_INPUT_SIZE 112 
#define FC1_OUTPUT_SIZE 128
#define FC2_OUTPUT_SIZE 5
// #define DEBUG 0

main(){

    printf("Boot... \n\n");


    // Create Fake input
    float input_vector[INPUT_SIZE];
    int i = 0;
    for (i = 0; i < INPUT_SIZE; i++)
    {
        input_vector[i] = i+1;
    }

    
    // int val = sizeof(input_vector)/sizeof(input_vector[0]);
    // int val = sizeof(input_vector)/sizeof(input_vector[0]);
    // printf("inputv %d", val);


/////////////////////////////////////////////////////////////////////////////////////// LAYER 1
// CONVOLUTION 1
    float conv0_currentKernel[KERNEL_SIZE];
    float conv0_current_bias = 0;
    float conv0_featureMap[NUM_FILTERS][CONV0_INPUT_SIZE-4];
    printf("\n\n Conv0 #################################################################3");

    for (int k = 0; k < NUM_FILTERS; k++)
    {
        // Load Current Weights
        // printf("\n\nCurrKernel = |");
        for ( i = 0; i < KERNEL_SIZE; i++)
        {
            conv0_currentKernel[i] = conv0_weights[i + (k * KERNEL_SIZE)];
            // printf("%f|",conv1_currentKernel[i]);
        }
        // printf("\n");

        // Load Current Bias
        conv0_current_bias = conv0_bias[k];
        


        // Perform Kernel operation
        for (i = 0; i <= sizeof(input_vector)/sizeof(input_vector[0])-KERNEL_SIZE; i++)
        {
            float totalSum = 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
            //    printf("j=%d | i=%d\n",j,i);
               totalSum += input_vector[i+j] * conv0_currentKernel[j];
            }
            // printf("Finished Sum=%f  k=%d  i=%d\n\n ",totalSum,k,i);

            conv0_featureMap[k][i] = totalSum + conv0_current_bias;

            
        }
    }
///// RELU

    // printf("\nsizeof=%d",sizeof(conv0_featureMap));
    // printf("\nsizeof=%d",sizeof(conv0_featureMap[0])/sizeof(conv0_featureMap[0][0]));
    // printf("\nsizeof=%d",sizeof(conv0_featureMap[0][0]));
    
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        // printf("\n\ni=%d",i);
        for (int j = 0; j < sizeof(conv0_featureMap[0])/sizeof(conv0_featureMap[0][0]) ; j++)
        {
            // printf("i=%d | j=%d\n",i,j);
            if (conv0_featureMap[i][j] <= 0)
                conv0_featureMap[i][j] = 0;
        }
    }


   

/////////////////////////////////////////////////////////////////////////////////////// LAYER 3
// CONVOLUTION 3
    float conv3_currentKernel[KERNEL_SIZE];
    float conv3_current_bias = 0;
    float conv3_featureMap[NUM_FILTERS][CONV3_INPUT_SIZE-4]; // 112
    printf("\n\n Conv3   ###############################################################3");


    // NAO TEM SEG FAULT ATE AQUI

    for (int k = 0; k < NUM_FILTERS; k++)
    {
        // Load Current Weights
        for ( i = 0; i < KERNEL_SIZE; i++)
        {
            conv3_currentKernel[i] = conv3_weights[i + (k * KERNEL_SIZE)];
            // printf("%f|",conv2_currentKernel[i]);
        }
        // printf("\n");

        // Load Current Bias
        conv3_current_bias = conv3_bias[k];
        // printf("CurrBias=%f",conv3_current_bias);

        

        // Perform Kernel operation
        for (i = 0; i <= sizeof(conv0_featureMap[0])/sizeof(conv0_featureMap[0][0])-KERNEL_SIZE; i++)
        {
            float totalSum = 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
            //    printf("j=%d | i=%d\n",j,i);
               totalSum += conv0_featureMap[k][i+j] * conv3_currentKernel[j];
               
            }
            // printf("Finished Sum=%f\n\n",totalSum);

            conv3_featureMap[k][i] = totalSum + conv3_current_bias;
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

    
    

/////////////////////////////////////////////////////////////////////////////////////// LAYER 6
// CONVOLUTION 6

    float conv6_currentKernel[KERNEL_SIZE];
    float conv6_current_bias = 0;
    float conv6_featureMap[NUM_FILTERS][CONV6_INPUT_SIZE-4]; // 112
    printf("\n\n Conv6  ##########################################################");


    // NAO TEM SEG FAULT ATE AQUI

    for (int k = 0; k < NUM_FILTERS; k++)
    {
        // Load Current Weights
        for ( i = 0; i < KERNEL_SIZE; i++)
        {
            conv6_currentKernel[i] = conv6_weights[i + (k * KERNEL_SIZE)];
            // printf("%f|",conv2_currentKernel[i]);
        }
        // printf("\n");

        // Load Current Bias
        conv6_current_bias = conv3_bias[k];
        // printf("CurrBias=%f",conv3_current_bias);

        

        // Perform Kernel operation
        for (i = 0; i <= sizeof(conv3_featureMap[0])/sizeof(conv3_featureMap[0][0])-KERNEL_SIZE; i++)
        {
            float totalSum = 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
            //    printf("j=%d | i=%d\n",j,i);
               totalSum += conv3_featureMap[k][i+j] * conv6_currentKernel[j];
               
            }
            // printf("Finished Sum=%f\n\n",totalSum);

            conv6_featureMap[k][i] = totalSum + conv6_current_bias;
        }
    }
    
    
///// RELU
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        for (int j = 0; j < CONV3_INPUT_SIZE-4; j++)
        {
            if (conv6_featureMap[i][j] <= 0)
                conv6_featureMap[i][j] = 0;            
        }
    }


////////// PRINT FEATUREMAP
    // printf("\n\nFeatureMaps  ");
    // for (int k = 0; k < NUM_FILTERS; k++)
    // {
    //     printf("------- Filter = %d\n",k);
    //     for (int i = 0; i < INPUT_SIZE-4; i++)
    //     {
    //         printf("%f|",conv1_featureMap[k][i]);
    //     }
    //     printf("\n");

    // }
/////////////////////////////////////////////////////////////////////////////////////////// LAYER 2
////////////////////////////////////////// FC Layer
///// Flatten

    float flatten1_vector[NUM_FILTERS*(CONV6_INPUT_SIZE-4)];
    int index = 0;

    printf("\n\n Flatten 1  ######################################################################");
    for (int k = 0; k < NUM_FILTERS; k++)
    {
        for (int i = 0; i < CONV6_INPUT_SIZE-4; i++)
        {
            // printf("k=%d | i=%d\n",k,i);
            flatten1_vector[index] =  conv6_featureMap[k][i];
            index++;
        }
    }
    
/////// FC 1 (6192 > 128)

    float fc1_out_vector[FC1_OUTPUT_SIZE];
    float totalValue;
    int inputVectorSize = NUM_FILTERS*(CONV6_INPUT_SIZE-4);

    printf("\n\n FC 1  ########################################################################3");

    for (int outputIndex = 0; outputIndex < FC1_OUTPUT_SIZE; outputIndex++){
        totalValue = 0;
        for (int i = 0; i < inputVectorSize; i++)
        {
            totalValue += flatten1_vector[i] * fc1_weights[(inputVectorSize*outputIndex)+i];
        }
        fc1_out_vector[outputIndex] = totalValue + fc1_bias[outputIndex];
    }



/////// FC 1 (128 > 5)

    float fc2_out_vector[FC2_OUTPUT_SIZE];
    float fc2_totalValue;

    printf("\n\n FC 2 ################################################################### ");

    for (int outputIndex = 0; outputIndex < FC2_OUTPUT_SIZE; outputIndex++){
        fc2_totalValue = 0;
        for (int i = 0; i < FC1_OUTPUT_SIZE; i++)
        {
            fc2_totalValue += fc1_out_vector[i] * fc2_weights[(inputVectorSize*outputIndex)+i];
        }
        fc2_out_vector[outputIndex] = fc2_totalValue + fc2_bias[outputIndex];
    }


    
////// print
    for (int i = 0; i < inputVectorSize; i++){
        // printf("%f\n",flatten1_vector[i]);
    }
    printf("\n\n Result Classes\n");
    for (int i = 0; i < 5; i++){
        printf("%f",fc1_out_vector[i]);
        printf("\n");
    }








    printf("\n\n ------------------------------------------------------------- \nEnd..");
    return 0;
}
