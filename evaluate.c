#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VALUES 10000
#define BATCH_SIZE 32
#define DATASET_UNITS 3923

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




int main() {
    FILE *file;
    char filename[] = "results.txt";
    char buffer[100000];
    int values[MAX_VALUES];
    int count = 0;

    // Abrir o arquivo
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        return 1;
    }

    // Ler os valores do arquivo
    fgets(buffer, sizeof(buffer), file);

    char *token = strtok(buffer, ",");
    while (token != NULL && count < MAX_VALUES) {
        values[count++] = atoi(token);
        token = strtok(NULL, ",");
    }

    // Fechar o arquivo
    fclose(file);


    int batchSizeAuxCount = 0;
    int batchCurrentAdd[5][5];
    int confusionMatrix[5][5];
    memset(batchCurrentAdd, 0, sizeof(batchCurrentAdd));
    memset(confusionMatrix, 0, sizeof(confusionMatrix));
    int predictedList[DATASET_UNITS];
    int expectedList[DATASET_UNITS];
    int correctLabels = 0;
    int wrongLabels = 0;
    int listIndex = 0;
    // Imprimir os valores lidos
    for (int i = 0; i < count-1; i = i+2) {
        int predicted = values[i];
        int correct = values[i+1];
        // printf("%d  %d", values[i], values[i+1]);
        // printf("\n");



        if ( correct == predicted){
            // printf("Correct prediction (predicted %d) (correct %d)\n",predicted,correct);
            if (batchCurrentAdd[correct][predicted] == 0){
                correctLabels++;
                confusionMatrix[correct][predicted]++;
                batchCurrentAdd[correct][predicted] = 1;
            }
        } else {
            // printf("Wrong prediction  (predicted %d) (correct %d)\n",predicted,correct);
            if (batchCurrentAdd[correct][predicted] == 0){
                wrongLabels++;
                confusionMatrix[correct][predicted]++;
                batchCurrentAdd[correct][predicted] = 1;
            }
        }
        predictedList[listIndex] = predicted;
        expectedList[listIndex] = correct;
        listIndex++;


        /// Update batch size parameters
        if (batchSizeAuxCount == BATCH_SIZE-1){
            memset(batchCurrentAdd, 0, sizeof(batchCurrentAdd));
            batchSizeAuxCount = 0;
        } else {
            batchSizeAuxCount += 1;
        }
    }
    printf("\n");
    printf("TOTAL SAMPLES: %d\n",count/2);


    printf("----------------------\n");
    printf("Accuracy: %0.1f\n",(float)correctLabels/((float)correctLabels + (float)wrongLabels)*100);
    printf("-----------------------\n");
    printf("Correct predictions: %d \n",correctLabels);
    printf("Wrong predictions: %d \n",wrongLabels);
    printf("----------------------\n");
    printf("Confusion Matrix\n");
    print_confusion_matrix(confusionMatrix);

    return 0;
}
