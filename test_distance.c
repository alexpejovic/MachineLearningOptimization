#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <string.h>
#include "knn.h"

/* A simple program to test the cosine distance function
 * On teach.cs, the following call produces the results below
 * ./test_distance /u/csc209h/winter/pub/datasets/a2_datasets/testing_data.bin
 * Cosine distance = 0.900966
 * Euclidean distance = 3205.300298
 */

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "Usage: %s filename\n", argv[0]);
        exit(1);
    }

    Dataset *data = load_dataset(argv[1]);

    // Compute the distance between the first two images in the data set

    double cos_distance = distance_cosine(&data->images[0], &data->images[1]);
    double euc_distance = distance_euclidean(&data->images[0], &data->images[1]);

    printf("Cosine distance = %f\n", cos_distance);
    printf("Euclidean distance = %f\n", euc_distance);
    return 0;
}
