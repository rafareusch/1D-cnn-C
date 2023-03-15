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



main(){

    printf("Boot... \n\n");


    // Create Fake input
    float input_vector[INPUT_SIZE];
    int i = 0;
    for (i = 0; i < INPUT_SIZE; i++)
    {
        input_vector[i] = i+1;
    }

    
    


/////////////////////////////////////////////////////////////////////////////////////// LAYER 1
// CONVOLUTION 1
    float conv1_currentKernel[KERNEL_SIZE];
    float current_bias = 0;
    float conv1_featureMap[NUM_FILTERS][INPUT_SIZE-4];


    for (int k = 0; k < NUM_FILTERS; k++)
    {
        // Load Current Weights
        // printf("\n\nCurrKernel = |");
        for ( i = 0; i < KERNEL_SIZE; i++)
        {
            conv1_currentKernel[i] = in_2[i + (k * KERNEL_SIZE)];
            // printf("%f|",conv1_currentKernel[i]);
        }
        // printf("\n");

        // Load Current Bias
        current_bias = in_1[k];
        // printf("CurrBias=%f",current_bias);


        // Perform Kernel operation
        for (i = 0; i <= INPUT_SIZE-KERNEL_SIZE; i++)
        {
            float totalSum = 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
            //    printf("j=%d | i=%d\n",j,i);
               totalSum += input_vector[i+j] * conv1_currentKernel[j];
            }
            // printf("Finished Sum=%f\n\n",totalSum);

            conv1_featureMap[k][i] = totalSum + current_bias;
        }
    }
    
    
///// RELU
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        for (int j = 0; j < INPUT_SIZE-4; j++)
        {
            if (conv1_featureMap[i][j] <= 0)
                conv1_featureMap[i][j] = 0;            
        }
    }
    
    

// ////////// PRINT FEATUREMAP
//     printf("\n\nFeatureMaps  ");
//     for (int k = 0; k < NUM_FILTERS; k++)
//     {
//         printf("------- Filter = %d\n",k);
//         for (int i = 0; i < INPUT_SIZE-4; i++)
//         {
//             printf("%f|",conv1_featureMap[k][i]);
//         }
//         printf("\n");

//     }
///////////////////////////////////////////////////////////////////////////////////////////// LAYER 2
//////////////////////////////////////////// FC Layer
/////// Flatten

    float flatten1_vector[NUM_FILTERS*(INPUT_SIZE-4)];
    int index = 0;

    printf("\n\n Flatten 1  ");
    for (int k = 0; k < NUM_FILTERS; k++)
    {
        for (int i = 0; i < INPUT_SIZE-4; i++)
        {
            flatten1_vector[index] =  conv1_featureMap[k][i];
            index++;
        }
    }
    
/////// FC 
    float fc1_out_vector[5];
    float totalValue;
    int inputVectorSize = NUM_FILTERS*(INPUT_SIZE-4);

    printf("\n\n FC 1  ");
    printf("inputinputVectorSize = %d ",inputVectorSize);
    for (int outputIndex = 0; outputIndex < 5; outputIndex++){
        totalValue = 0;
        for (int i = 0; i < inputVectorSize; i++)
        {
            totalValue += flatten1_vector[i] * in_10[(inputVectorSize*outputIndex)+i];
        }
        fc1_out_vector[outputIndex] = totalValue + in_7[outputIndex];
    }



    
////// print
    for (int i = 0; i < inputVectorSize; i++){
        printf("%f\n",flatten1_vector[i]);
    }
    printf("\n\n Result Classes\n");
    for (int i = 0; i < 5; i++){
        printf("%f",fc1_out_vector[i]);
        printf("\n");
    }


// /////////////////////////////////////////////////////////////////////////////// LAYER 2
// //////// Convolution 2


//     for (int k = 0; k < NUM_FILTERS; k++)
//     {
//         // Load Current Weights
//         // printf("\n\nCurrKernel = |");
//         for ( i = 0; i < KERNEL_SIZE; i++)
//         {
//             conv1_currentKernel[i] = in_2[i + (k * KERNEL_SIZE)];
//             // printf("%f|",conv1_currentKernel[i]);
//         }

//         // Load Current Bias
//         current_bias = in_1[k];
//         // printf("CurrBias=%f",current_bias);


//         // Perform Kernel operation
//         for (i = 0; i <= INPUT_SIZE-KERNEL_SIZE; i++)
//         {
//             float totalSum = 0;
//             for (int j = 0; j < KERNEL_SIZE; j++)
//             {
//             //    printf("j=%d | i=%d\n",j,i);
//                totalSum += input_vector[i+j] * conv1_currentKernel[j];
//             }
//             // printf("Finished Sum=%f\n\n",totalSum);

//             featuremaps[k][i] = totalSum + current_bias;
//         }
//     }






    printf("\n\n ------------------------------------------------------------- \nEnd..");
    return 0;
}
