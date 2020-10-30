#include "utility.h"
#include <string.h>
#include <cstdlib>

// for calculating median
#include <math.h>
#include <limits>  // for using limits for min, max values
#include <algorithm>  // std::sort
#include <vector>


char *char_concat(char *c1, const char *c2){  //concatenate two strings with dynamic allocation
  char *concatenated;
  if(!c1){
    concatenated=(char*)malloc(sizeof(char)*(strlen(c2)+1));
    strcpy(concatenated,c2);
  }
  else{
    concatenated=(char*)realloc(c1,sizeof(char)*(strlen(c1)+strlen(c2)+1));
    strcat(concatenated,c2);
  }
  return concatenated;
}


double randDouble(double min, double max){
    double f = (double)rand() / RAND_MAX;
    return min + f*(max - min);
}

int randInt(int min, int max){
    return (rand() % (max-min+1)) + min;
}

void get_min_median_max_double(double *min_fit, double *median, double *max_fit, int *count_valid, int *n_discarded, double *pool, int size, int invalid_val) {
    *min_fit = std::numeric_limits<double>::max();
    *max_fit = std::numeric_limits<double>::min();
    std::vector<double> vec_fitnesses;
    *n_discarded = 0;
    *count_valid = 0;
    for(int i = 0; i < size; i++) {
        if(pool[i] != invalid_val) {
            vec_fitnesses.push_back(pool[i]);
            *max_fit = fmax(pool[i], *max_fit);
            *min_fit = fmin(pool[i], *min_fit);
            *count_valid += 1;
        } else {
            *n_discarded += 1;
        }
    }

    std::sort(vec_fitnesses.begin(), vec_fitnesses.end());

    // now get median value
    if((*count_valid % 2) == 0) {
        *median = (
            vec_fitnesses[(int)(*count_valid / 2)] +
            vec_fitnesses[(int)((*count_valid / 2) + 1)]
        ) / 2;
    } else {
        *median = vec_fitnesses[(int)(*count_valid / 2)];
    }
}

void get_min_median_max_int(int *min_fit, int *median, int *max_fit, int *count_valid, int *n_discarded, int *pool, int size, int invalid_val) {
    *min_fit = std::numeric_limits<int>::max();
    *max_fit = std::numeric_limits<int>::min();
    std::vector<int> vec_fitnesses;
    *n_discarded = 0;
    *count_valid = 0;
    for(int i = 0; i < size; i++) {
        if(pool[i] != invalid_val) {
            vec_fitnesses.push_back(pool[i]);
            *max_fit = fmax(pool[i], *max_fit);
            *min_fit = fmin(pool[i], *min_fit);
            *count_valid += 1;
        } else {
            *n_discarded += 1;
        }
    }

    std::sort(vec_fitnesses.begin(), vec_fitnesses.end());

    // now get median value
    if((*count_valid % 2) == 0) {
        *median = (
            vec_fitnesses[(int)(*count_valid / 2)] +
            vec_fitnesses[(int)((*count_valid / 2) + 1)]
        ) / 2;
    } else {
        *median = vec_fitnesses[(int)(*count_valid / 2)];
    }
}