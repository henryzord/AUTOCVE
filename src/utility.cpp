#include "utility.h"
#include <string.h>
#include <cstdlib>

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

int compare (const void * a, const void * b) {
  return (*(int*)a - *(int*)b);
}

double *getCurve(double *predictions, int n_instances, int n_classes, int classIndex) {
    if ((predictions == NULL) || (n_instances == 0) || (n_classes <= classIndex)) {
      return NULL;
    }

    double totPos = 0, totNeg = 0;

    double *classPreds = (double*)malloc(sizeof(double) * n_instances);

    // Get distribution of positive/negatives
    for (int i = 0; i < n_instances; i++) {
      double pred = predictions[(i * n_classes) + classIndex];
      classPreds[i] = pred;

      double other_max = pred;
      int max_index = classIndex;
      for(int j = 0; j < n_classes; j++) {
        if(predictions[(i * n_classes) + j] > other_max) {
            other_max = predictions[(i * n_classes) + j];
            max_index = j;
        }
      }

      if(max_index == classIndex) {
        totPos += 1;
      } else {
        totNeg += 1;
      }
    }

    qsort(classPreds, n_instances, sizeof(double), compare);  // TODO parei aqui

    return classPreds;  // TODO wrong, change later

//    TwoClassStats tc = new TwoClassStats(totPos, totNeg, 0, 0);
//    double threshold = 0;
//    double cumulativePos = 0;
//    double cumulativeNeg = 0;
//
//    for (int i = 0; i < sorted.length; i++) {
//
//      if ((i == 0) || (probs[sorted[i]] > threshold)) {
//        tc.setTruePositive(tc.getTruePositive() - cumulativePos);
//        tc.setFalseNegative(tc.getFalseNegative() + cumulativePos);
//        tc.setFalsePositive(tc.getFalsePositive() - cumulativeNeg);
//        tc.setTrueNegative(tc.getTrueNegative() + cumulativeNeg);
//        threshold = probs[sorted[i]];
//        insts.add(makeInstance(tc, threshold));
//        cumulativePos = 0;
//        cumulativeNeg = 0;
//        if (i == sorted.length - 1) {
//          break;
//        }
//      }
//
//      NominalPrediction pred = (NominalPrediction) predictions.get(sorted[i]);
//
//      if (pred.actual() == Prediction.MISSING_VALUE) {
//        System.err.println(getClass().getName()
//          + " Skipping prediction with missing class value");
//        continue;
//      }
//      if (pred.weight() < 0) {
//        System.err.println(getClass().getName()
//          + " Skipping prediction with negative weight");
//        continue;
//      }
//      if (pred.actual() == classIndex) {
//        cumulativePos += pred.weight();
//      } else {
//        cumulativeNeg += pred.weight();
//      }
//
//    }
//
//    // make sure a zero point gets into the curve
//    if (tc.getFalseNegative() != totPos || tc.getTrueNegative() != totNeg) {
//      tc = new TwoClassStats(0, 0, totNeg, totPos);
//      threshold = probs[sorted[sorted.length - 1]] + 10e-6;
//      insts.add(makeInstance(tc, threshold));
//    }
//
//    free(classPreds);  // TODO maybe reuse for insts?
//
//    return insts;
}
