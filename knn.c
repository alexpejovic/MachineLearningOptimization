#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "knn.h"

/****************************************************************************/
/* For all the remaining functions you may assume all the images are of the */
/*     same size, you do not need to perform checks to ensure this.         */
/****************************************************************************/
/**
 * load_dataset takes the name of the binary file containing the data and
 * loads it into memory. The binary file format consists of the following:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - 784 bytes : Image 1 data (WIDTHxWIDTH)
 *          ...
 *     -   1 byte  : Image N label
 *     - 784 bytes : Image N data (WIDTHxWIDTH)
 *
 * If the filename does not exist then the function will return a NULL pointer.
 */
Dataset *load_dataset(const char *filename) {
    Dataset *data = malloc(sizeof(Dataset));

    FILE *f = fopen(filename, "rb");
    if(f == NULL) {
        return NULL;
    }
    if(fread(&data->num_items, sizeof(int), 1, f)==-1){
        fprintf(stderr, "Could not read num items from %s\n", filename);
        exit(1);
    }

    data->labels = malloc(sizeof(unsigned char) * data->num_items);
    data->images = malloc(sizeof(Image) * data->num_items);

    for (int i = 0; i < data->num_items; i++) {
        if(fread(data->labels + i, sizeof(unsigned char), 1, f) != 1) {
            fprintf(stderr, "Error: expecting to read a label from %s\n", filename);
            exit(1);
        }
        data->images[i].sx = WIDTH;
        data->images[i].sy = WIDTH;

        data->images[i].data = malloc(sizeof(unsigned char) * NUM_PIXELS);
        if(fread(data->images[i].data, sizeof(unsigned char), NUM_PIXELS, f) != NUM_PIXELS) {
            fprintf(stderr, "Error: expecting to read the pixels from image %d\n", i);
            exit(1);
        }
    }
    if(fclose(f) != 0) {
        perror("fclose");
        exit(1);
    }
    return data;
}


/**
 * Return the euclidean distance between the image pixels (as vectors).
 * Specifically  d = sqrt( sum((a[i]-b[i])^2))
 */
double distance_euclidean(Image *a, Image *b) {
    double d = 0;
    for (int i = 0; i < a->sx * a->sy; i++) {
        d += (a->data[i] - b->data[i]) * (a->data[i] - b->data[i]);
    }
    return sqrt(d);
}

typedef struct {
    double dist;
    int img_idx;
} Knn_item;

/**
 * Given the input training dataset, an image to classify and K as well as a
 * distance function specified by fptr,
 *   (1) Find the K most similar images to `input` in the dataset
 *   (2) Return the most frequent label of these K images.  If two are tied,
 *       output the smaller label.
 */
int knn_predict(Dataset *data, Image *input, int K, double (*fptr)(Image *, Image *)) {

    // Array to keep track of K-closest images so far.
    Knn_item smallest[K];
    for (int i = 0; i < K; i++) {
        smallest[i].dist = INFINITY;
    }
    // For each training image, compute the distance using the function pointer
    for (int i = 0; i < data->num_items; i++) {

        // TODO Change the call below to call distance function passed in as
        // a parameter
        double dist = fptr(&data->images[i], input);

        // Find the maximum distance among the previous K closest
        double max_dist = -1;
        int max_index = 0;
        for (int j = 0; j < K; j++) {
            if (smallest[j].dist > max_dist) {
                max_dist = smallest[j].dist;
                max_index = j;
            }
        }

        // If current distance one of K-closest so far, update the values.
        if (dist < max_dist) {
            smallest[max_index].dist = dist;
            smallest[max_index].img_idx = i;
        }
    }

    // Count the frequencies of the labels
    int counts[10] = {0};
    for (int i = 0; i < K; i++) {
        counts[data->labels[smallest[i].img_idx]]++;
    }

    // Find the most frequent label
    int max_count = 0, max_label = 1;
    for (int i = 0; i < 10; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_label = i;
        }
    }

    return max_label;
}

/**
 * Free all the allocated memory for the dataset
 * Check to ensure that the function works properly when `data' is allocated
 * as well as when `data' is not allocated.
 */
void free_dataset(Dataset *data) {
    if (data == NULL) {
        return;
    }

    for (int i = 0; i < data->num_items; i++) {
        free(data->images[i].data);
    }
    free(data->images);
    free(data->labels);
    free(data);
}



/************************** A3 Code below ************************************/

/**
 * child_handler will be called by each child process, and is where the
 * kNN predictions happen. Along with the training and testing datasets, the
 * function also takes in
 *    (1) File descriptor for a pipe with input coming from the parent: p_in
 *    (2) File descriptor for a pipe with output going to the parent:  p_out
 *
 * Once this function is called, the child should do the following:
 *    - Read an integer `start_idx` from the parent (through p_in)
 *    - Read an integer `N` from the parent (through p_in)
 *    - Call `knn_predict()` on testing images `start_idx` to `start_idx+N-1`
 *    - Write an integer representing the number of correct predictions to
 *        the parent (through p_out)
 */
void child_handler(Dataset *training, Dataset *testing, int K,
                   double (*fptr)(Image *, Image *),int p_in, int p_out) {

    int start_idx;
    int N;
    int correct_predictions = 0;

    if(read(p_in, &start_idx, sizeof(int)) < 0) {
      perror("Reading start_idx in knn");
      exit(1);
    }

    if(read(p_in, &N, sizeof(int)) < 0) {
      perror("Reading N in knn");
      exit(1);
    }

    int prediction;
    int label;

    for(int i = start_idx; i < start_idx + N; i++){
      prediction = knn_predict(training, &(testing->images[i]), K, fptr);
      label = (int)testing->labels[i];

      if(prediction == label){
        ++correct_predictions;
      }
    }

    if(write(p_out, &correct_predictions, sizeof(int)) < 0){
      perror("Writing correct_predictions in knn");
      exit(1);
    }
}

/**
 * This function computes the cosine distance.  It should be called similarly to
 * the function distance() above except the formula that it should evaluate is
 * distance = 2*arccos( sum(a[i]*b[i]) / (sqrt(sum(a[i]^2) sqrt(sum(b[i]^2)))/pi
 * See (https://en.wikipedia.org/wiki/Cosine_similarity) for more information.
 *   - use the constant M_PI for pi.  M_PI is defined in math.h
 *   - "man acos" describes the arc cos funciton in the C math library
*/
double distance_cosine(Image *a, Image *b){

    double ab_sum = 0;
    double asq_sum = 0;
    double bsq_sum = 0;
    double ai;
    double bi;

    for (int i = 0; i < a->sx * a->sy; i++) {
      ai = a->data[i];
      bi = b->data[i];
      ab_sum += ai * bi;
      asq_sum += ai * ai;
      bsq_sum += bi * bi;
    }

    return 2 * acos(ab_sum / (sqrt(asq_sum)  * sqrt(bsq_sum))) / M_PI;
}
