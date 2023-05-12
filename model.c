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


//// Main defines
#include "params/dataset120_eval.h" // 3923 
#define BATCH_SIZE 32
#define DATASET_UNITS 3923
#define INPUT_SIZE 120
#define NUM_FILTERS 64
#define KERNEL_SIZE 5
#define CONV0_INPUT_SIZE 120 
#define CONV3_INPUT_SIZE 116
#define CONV6_INPUT_SIZE 112 
#define FC1_OUTPUT_SIZE 128
#define FC2_OUTPUT_SIZE 5


// INT CNN defines
#define INPUT_MULTIP 1000 // This value MUST be the same used in Quantization
#define CALCULATE_FLOAT 0  // 1 = No Quantization     0 = Quantization


void print_confusion_matrix(int matrix[5][5]) {
    int i = 0;
    int j = 0;
    int n = 5;

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
    int batchSizeAuxCount = 0;
    int batchCurrentAdd[5][5];
    int confusionMatrix[5][5];
    memset(batchCurrentAdd, 0, sizeof(batchCurrentAdd));
    memset(confusionMatrix, 0, sizeof(confusionMatrix));
    int predictedList[DATASET_UNITS];
    int expectedList[DATASET_UNITS];
    int listIndex = 0;
    printf("Evaluating...\n");



    for(int datasetIndex = 0 ; datasetIndex < DATASET_UNITS ; datasetIndex++ ){

        // DATASET READING AND PREPARATION
        float progress = datasetIndex / (float)DATASET_UNITS;
        printf("Progress %0.f\n",progress*100);
        progress = (datasetIndex/DATASET_UNITS)*100;

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
        int INTconv0_featureMap[NUM_FILTERS][CONV0_INPUT_SIZE-4];
        int INTconv0_currentKernel[KERNEL_SIZE];
        int INTconv0_current_bias = 0;

        //printf(" Conv0 #################################################################\n");

        for (int k = 0; k < NUM_FILTERS; k++)
        {
            // Load Current Weights
            for (i = 0; i < KERNEL_SIZE; i++)
            {
                if(CALCULATE_FLOAT == 1)conv0_currentKernel[i] = conv0_weights[i + (k * KERNEL_SIZE)];
                INTconv0_currentKernel[i] = (int) (conv0_weights_lut[conv0_weights_indices[i + (k * KERNEL_SIZE)]]);
            }

            // Load Current Bias
            if(CALCULATE_FLOAT == 1)conv0_current_bias = conv0_bias[k];
            INTconv0_current_bias = (int) (conv0_bias_lut[conv3_bias_indices[k]]);


            // Perform Kernel operation
            for (i = 0; i <= sizeof(input_vector)/sizeof(input_vector[0])-KERNEL_SIZE; i++)
            {
                float totalSum = 0;
                float INTtotalSum = 0;

                for (int j = 0; j < KERNEL_SIZE; j++)
                {
                if(CALCULATE_FLOAT == 1)totalSum += input_vector[i+j] * conv0_currentKernel[j];
                INTtotalSum += ((int)( (input_vector[i+j] + (float)(5/(float)INPUT_MULTIP) ) * INPUT_MULTIP)) * INTconv0_currentKernel[j];
                }
                if(CALCULATE_FLOAT == 1)conv0_featureMap[k][i] = totalSum + conv0_current_bias;
                INTconv0_featureMap[k][i] = INTtotalSum + INTconv0_current_bias;
            }
        }

        
    //////////////////////////////// INT HANDLER
    // divide the feature map items by INPUT_MULTIP
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV0_INPUT_SIZE-4; j++)
            {
                // printf("%2.8f %2.8f\n",conv0_featureMap[0][j], ((float) (INTconv0_featureMap[0][j]))/(INPUT_MULTIP) ); 
                INTconv0_featureMap[i][j] = (INTconv0_featureMap[i][j])/(INPUT_MULTIP);
                // printf("---  FLOAT ------------------  INT ------\n");
                // printf("%2.8f %2.8f\n",conv0_featureMap[0][j], ((float) (INTconv0_featureMap[0][j]))/(INPUT_MULTIP*INPUT_MULTIP) ); // igual
            }
        }




    ///// RELU
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV0_INPUT_SIZE-4; j++)
            {
                if(CALCULATE_FLOAT == 1){
                    if (conv0_featureMap[i][j] <= 0)
                        conv0_featureMap[i][j] = 0;
                }
                if (INTconv0_featureMap[i][j] <= 0)
                    INTconv0_featureMap[i][j] = 0;     
            }
        }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 3
    // CONVOLUTION 3
        float conv3_featureMap[NUM_FILTERS][CONV3_INPUT_SIZE-4]; // 112
        float conv3_totalSum = 0;

        int INTconv3_featureMap[NUM_FILTERS][CONV3_INPUT_SIZE-4]; // 112
        int INTconv3_totalSum = 0;

        for (int filterToGenerate = 0 ; filterToGenerate < NUM_FILTERS ; filterToGenerate ++ ){
            for (int inputOffset = 0 ; inputOffset < CONV3_INPUT_SIZE-4 ; inputOffset++){
                conv3_totalSum = 0;
                INTconv3_totalSum = 0;
                for (int filterIn = 0 ; filterIn < NUM_FILTERS ; filterIn++){
                    for (int kernelIndex = 0 ; kernelIndex < KERNEL_SIZE ; kernelIndex++){
                        int weightIndex = kernelIndex + (filterIn * KERNEL_SIZE) + ( filterToGenerate * NUM_FILTERS * KERNEL_SIZE ) ;
                        int indexIn = kernelIndex + (inputOffset);

                        if(CALCULATE_FLOAT == 1)conv3_totalSum += conv0_featureMap[filterIn][indexIn] * conv3_weights[weightIndex]; 
                        INTconv3_totalSum += ((INTconv0_featureMap[filterIn][indexIn] * INPUT_MULTIP) / INPUT_MULTIP)    *  ((int) (conv3_weights_lut[conv3_weights_indices[weightIndex]]))  ; 
                    }
                }
                if(CALCULATE_FLOAT == 1)conv3_totalSum += conv3_bias[filterToGenerate];
                if(CALCULATE_FLOAT == 1)conv3_featureMap[filterToGenerate][inputOffset] = conv3_totalSum;

                INTconv3_totalSum += ( (int) conv3_bias_lut[conv3_bias_indices[filterToGenerate]]);
                INTconv3_featureMap[filterToGenerate][inputOffset] = INTconv3_totalSum;
            }
        }
        
        

    //////////////////////////////// INT HANDLER
    // divide the feature map items by INPUT_MULTIP
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV3_INPUT_SIZE-4; j++)
            {            
                // printf("conv3---------------\n");
                // printf("%2.8f %2.8f\n",conv3_featureMap[i][j], ((float) (INTconv3_featureMap[i][j]))/(INPUT_MULTIP*INPUT_MULTIP) );
                // printf("%2.8f %2.8f\n",conv3_featureMap[i][j], ((float) (INTconv3_featureMap[i][j]))/(INPUT_MULTIP) );
                INTconv3_featureMap[i][j] = (INTconv3_featureMap[i][j])/(INPUT_MULTIP);
            }
        }


        
    ///// RELU
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV3_INPUT_SIZE-4; j++)
            {
                if(CALCULATE_FLOAT == 1) { 
                    if (conv3_featureMap[i][j] <= 0)
                        conv3_featureMap[i][j] = 0;    
                }    
                if (INTconv3_featureMap[i][j] <= 0)
                    INTconv3_featureMap[i][j] = 0; 
            }
        }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 6
    // CONVOLUTION 6

        float conv6_featureMap[NUM_FILTERS][CONV6_INPUT_SIZE-4]; 
        float conv6_totalSum = 0;
        int INTconv6_featureMap[NUM_FILTERS][CONV6_INPUT_SIZE-4]; 
        int INTconv6_totalSum = 0;

        for (int filterToGenerate = 0 ; filterToGenerate < NUM_FILTERS ; filterToGenerate ++ ){
            for (int inputOffset = 0 ; inputOffset < CONV6_INPUT_SIZE-4 ; inputOffset++){
                conv6_totalSum = 0;
                INTconv6_totalSum = 0;
                for (int filterIn = 0 ; filterIn < NUM_FILTERS ; filterIn++){
                    for (int kernelIndex = 0 ; kernelIndex < KERNEL_SIZE ; kernelIndex++){
                        int weightIndex = kernelIndex + (filterIn * KERNEL_SIZE) + ( filterToGenerate * NUM_FILTERS * KERNEL_SIZE ) ;
                        int indexIn = kernelIndex + (inputOffset);
                        if(CALCULATE_FLOAT == 1)conv6_totalSum += conv3_featureMap[filterIn][indexIn] * conv6_weights[weightIndex]; 
                        INTconv6_totalSum += ((INTconv3_featureMap[filterIn][indexIn] * INPUT_MULTIP)/INPUT_MULTIP) * ((int) (conv6_weights_lut[conv6_weights_indices[weightIndex]]));
                    }
                }
                if(CALCULATE_FLOAT == 1)conv6_totalSum += conv6_bias[filterToGenerate];
                if(CALCULATE_FLOAT == 1)conv6_featureMap[filterToGenerate][inputOffset] = conv6_totalSum;
                INTconv6_totalSum += ( (int) conv6_bias_lut[conv6_bias_indices[filterToGenerate]]);
                INTconv6_featureMap[filterToGenerate][inputOffset] = INTconv6_totalSum;
            }
        }
        

    //////////////////////////////// INT HANDLER
    // divide the feature map items by INPUT_MULTIP
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV6_INPUT_SIZE-4; j++)
            {            
                // printf("conv6---------------\n");
                // printf("%2.8f %2.8f\n",conv6_featureMap[i][j], ((float) (INTconv6_featureMap[i][j]))/(INPUT_MULTIP*INPUT_MULTIP) );  //Out 0.500  0.0000500
                // printf("%2.8f %2.8f\n",conv6_featureMap[i][j], ((float) (INTconv6_featureMap[i][j]))/(INPUT_MULTIP) );         //Out 0.500  0.500
                INTconv6_featureMap[i][j] = (INTconv6_featureMap[i][j]/(INPUT_MULTIP)); // 5000
            }
        }

        
    /////////////////////////// RELU
        for (int i = 0; i < NUM_FILTERS; i++)
        {
            for (int j = 0; j < CONV6_INPUT_SIZE-4; j++)
            {
                if(CALCULATE_FLOAT == 1){ 
                    if (conv6_featureMap[i][j] <= 0)
                        conv6_featureMap[i][j] = 0; 
                }
                if (INTconv6_featureMap[i][j] <= 0)
                    INTconv6_featureMap[i][j] = 0; 
            }
        }







    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LAYER 2
    ////////////////////////////////////////// FC Layer
    ///// Flatten

        float flatten1_vector[NUM_FILTERS*(CONV6_INPUT_SIZE-4)];
        int INTflatten1_vector[NUM_FILTERS*(CONV6_INPUT_SIZE-4)];
        int index = 0;

        for (int k = 0; k < NUM_FILTERS; k++)
        {
            for (int i = 0; i < CONV6_INPUT_SIZE-4; i++)
            {
                if(CALCULATE_FLOAT == 1)flatten1_vector[index] =  conv6_featureMap[k][i];
                INTflatten1_vector[index] =  INTconv6_featureMap[k][i];
                index++;
            }
        }
        
    /////// FC 1 (6912 > 128)

        float fc1_out_vector[FC1_OUTPUT_SIZE];
        float totalValue;

        int INTfc1_out_vector[FC1_OUTPUT_SIZE];
        int INTtotalValue;
        int fc1_inputSize = NUM_FILTERS*(CONV6_INPUT_SIZE-4); // 6912


        for (int outputIndex = 0; outputIndex < FC1_OUTPUT_SIZE; outputIndex++){
            totalValue = 0;
            INTtotalValue = 0;
            for (int i = 0; i < fc1_inputSize; i++)
            {
                if(CALCULATE_FLOAT == 1)totalValue += flatten1_vector[i] * fc1_weights[(fc1_inputSize*outputIndex)+i];
                INTtotalValue += ((INTflatten1_vector[i]*INPUT_MULTIP)/INPUT_MULTIP) * ( (int) (fc1_weights_lut[fc1_weights_indices[(fc1_inputSize*outputIndex)+i]]) );
            }
            if(CALCULATE_FLOAT == 1)fc1_out_vector[outputIndex] = totalValue + fc1_bias[outputIndex];
            INTfc1_out_vector[outputIndex] = INTtotalValue + ((int) fc1_bias_lut[fc1_bias_indices[outputIndex]]);
        }


    //////////////////////////////// INT HANDLER
    // divide the feature map items by INPUT_MULTIP
        for (int i = 0; i < FC1_OUTPUT_SIZE; i++)
        {         
            // printf("fc1---------------\n");
            // printf("%2.8f %2.8f\n",fc1_out_vector[i], ((float) (INTfc1_out_vector[i]))/(INPUT_MULTIP*INPUT_MULTIP) );
            // printf("%2.8f %2.8f\n",fc1_out_vector[i], ((float) (INTfc1_out_vector[i]))/(INPUT_MULTIP) );
            INTfc1_out_vector[i] = INTfc1_out_vector[i] / INPUT_MULTIP;
        }



        
    ////////////////// RELU
        for (int i = 0; i < FC1_OUTPUT_SIZE ; i++){
            if(CALCULATE_FLOAT == 1) if ( fc1_out_vector[i] < 0 ) fc1_out_vector[i] = 0;        
            if ( INTfc1_out_vector[i] < 0 ) INTfc1_out_vector[i] = 0;        
        }



    ///////////////// FC 1 (128 > 5)
        float fc2_out_vector[FC2_OUTPUT_SIZE];
        float fc2_totalValue;
        int INTfc2_out_vector[FC2_OUTPUT_SIZE];
        int INTfc2_totalValue;
        
        for (int outputIndex = 0; outputIndex < FC2_OUTPUT_SIZE; outputIndex++){
            fc2_totalValue = 0;
            INTfc2_totalValue = 0;
            // if ( datasetIndex == 4)  printf("totalvalue------");
            for (int i = 0; i < FC1_OUTPUT_SIZE; i++)
            {
                if(CALCULATE_FLOAT == 1)fc2_totalValue += fc1_out_vector[i] * fc2_weights[(FC1_OUTPUT_SIZE*outputIndex)+i];
                INTfc2_totalValue += ((INTfc1_out_vector[i] * INPUT_MULTIP ) / INPUT_MULTIP) * ( (int) (fc2_weights_lut[fc2_weights_indices[(FC1_OUTPUT_SIZE*outputIndex)+i]])); 
                // if ( datasetIndex == 4) printf("%f  %d \n ",fc2_totalValue,INTfc2_totalValue);
            }
            if(CALCULATE_FLOAT == 1)fc2_out_vector[outputIndex] = fc2_totalValue + fc2_bias[outputIndex];
            INTfc2_out_vector[outputIndex] = INTfc2_totalValue + ( (int) fc2_bias_lut[fc2_bias_indices[outputIndex]] );
            // if ( datasetIndex == 4) printf("%f  %d \n ",fc2_out_vector[outputIndex],INTfc2_out_vector[outputIndex]);
        }


        
    //////////////////////////////// INT HANDLER
    // divide the feature map items by INPUT_MULTIP
        for (int i = 0; i < FC2_OUTPUT_SIZE; i++)
        {   
            INTfc2_out_vector[i] = INTfc2_out_vector[i] / (INPUT_MULTIP*INPUT_MULTIP);
            
        }




    ////////////////////////// Result Classes
        //printf("\n############################################################## RESULT CLASSES");

        int calculatedLabel;
        float maxValue;

        if (CALCULATE_FLOAT == 1){
            maxValue =  fc2_out_vector[0];
            calculatedLabel = 0;
            for (int i = 0; i < 5; i++){
                if (fc2_out_vector[i] > maxValue){
                    maxValue = fc2_out_vector[i];
                    calculatedLabel = i;
                }
            }
        } else {
            maxValue =  INTfc2_out_vector[0];
            calculatedLabel = 0;
            for (int i = 0; i < 5; i++){
                if (INTfc2_out_vector[i] > maxValue){
                    maxValue = INTfc2_out_vector[i];
                    calculatedLabel = i;
                }
            }
        }


        if ( targetLabel == calculatedLabel){
            printf("Correct prediction (predicted %d) (correct %d)\n",calculatedLabel,targetLabel);
            if (batchCurrentAdd[targetLabel][calculatedLabel] == 0){
                correctLabels++;
                confusionMatrix[targetLabel][calculatedLabel]++;
                batchCurrentAdd[targetLabel][calculatedLabel] = 1;
            }
        } else {
            printf("Wrong prediction  (predicted %d) (correct %d)\n",calculatedLabel,targetLabel);
            if (batchCurrentAdd[targetLabel][calculatedLabel] == 0){
                wrongLabels++;
                confusionMatrix[targetLabel][calculatedLabel]++;
                batchCurrentAdd[targetLabel][calculatedLabel] = 1;
            }
        }
        predictedList[listIndex] = calculatedLabel;
        expectedList[listIndex] = targetLabel;
        listIndex++;


        /// Update batch size parameters
        if (batchSizeAuxCount == BATCH_SIZE-1){
            memset(batchCurrentAdd, 0, sizeof(batchCurrentAdd));
            batchSizeAuxCount = 0;
        } else {
            batchSizeAuxCount += 1;
        }
    }


    printf("----------------------\n");
    printf("Accuracy: %0.1f\n",(float)correctLabels/((float)correctLabels + (float)wrongLabels)*100);
    printf("----------\n");
    printf("Correct predictions: %d \n",correctLabels);
    printf("Wrong predictions: %d \n",wrongLabels);
    printf("----------------------\n");
    printf("Confusion Matrix\n");
    print_confusion_matrix(confusionMatrix);
    printf(" \n------------------------------------------------------------- End..");
    return 0;
}