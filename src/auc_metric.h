#include <cstdlib>

struct pair {
    double value;
    int index;
} typedef pair_t;

struct roc_point {
    double tp;
    double fp;
    double threshold;
} typedef roc_point_t;

int compare (const void *a, const void *b) {
  return ((pair_t*)a)->value >= ((pair_t*)b)->value;
}

roc_point_t *getCurve(PyObject *y_pred, PyObject *y_true, int classIndex) {
    if(y_pred == NULL) {
        return NULL;
    }

    npy_intp *y_pred_dims = PyArray_DIMS((PyArrayObject*)y_pred);
    int n_instances = (int)y_pred_dims[0], n_classes = (int)y_pred_dims[1];

    if ((n_instances == 0) || (n_classes <= classIndex)) {
      return NULL;
    }

    npy_intp y_pred_itemsize = PyArray_ITEMSIZE((PyArrayObject*)y_pred);

    double totPos = 0, totNeg = 0;

    double *probs = (double*)malloc(sizeof(double) * n_instances);

    char *y_pred_ptr = PyArray_BYTES((PyArrayObject*)y_pred);
    // PyObject* iterator = PyArray_IterNew(y_pred);

    char isLastCorrect = -1;

    // Get distribution of positive/negatives
    double cur_pred, other_max;
    int max_index;
    for (int i = 0; i < n_instances; i++) {
        other_max = -1;
        max_index = -1;

        for(int j = 0; j < n_classes; j++) {
            cur_pred = PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)y_pred, y_pred_ptr));
            if(cur_pred > other_max) {
                other_max = cur_pred;
                max_index = j;
            }
            if(j == classIndex) {
                probs[i] = cur_pred;
            }
            y_pred_ptr += y_pred_itemsize;
        }

        if(max_index == classIndex) {
            totPos += 1;
            isLastCorrect = 1;
        } else {
            totNeg += 1;
            isLastCorrect = 0;
        }
    }

    pair_t *pairedArray = (pair_t*)malloc(sizeof(pair_t) * n_instances);
    for(int i = 0; i < n_instances; i++) {
        pairedArray[i] = {.value = probs[i], .index = i};
    }

    //     Actual Class
    //      0       1
    //    ---------------
    //   |       |       |
    // 0 |  TN   |  FN   | Predicted
    //   |       |       | Class
    //    ---------------
    //   |       |       |
    // 1 |  FP   |  TP   |
    //   |       |       |
    //    ---------------
    double threshold = 0;
    double cumulativePos = 0;
    double cumulativeNeg = 0;
    int tc[2][2] = {{0, 0}, {0, 0}};  // confusion matrix

    roc_point_t *points = (roc_point_t*)malloc(sizeof(roc_point_t) * n_instances);

    for(int i = 0; i < n_instances; i++) {
        if((i == 0) || probs[pairedArray[i].index] > threshold) {
            tc[1][1] = tc[1][1] - cumulativePos;
            tc[0][1] = tc[0][1] + cumulativePos;
            tc[1][0] = tc[1][0] - cumulativeNeg;
            tc[0][0] = tc[0][0] + cumulativeNeg;
            threshold = probs[pairedArray[i].index];
            points[i] = {.tp = tc[1][1], .fp = tc[1][0], .threshold = threshold};
            cumulativeNeg = 0;
            cumulativePos = 0;
            if(i == (n_instances - 1)) {
                break;
            }
        }

        if(isLastCorrect) {
            cumulativePos += 1;
        } else {
            cumulativeNeg += 1;
        }
    }

    // make sure a zero point gets into the curve
    if((tc[0][1] != totPos) || (tc[0][0] != totNeg)) {
        tc[1][1] = 0;
        tc[1][0] = 0;
        tc[0][0] = totNeg;
        tc[0][1] = totPos;
        threshold = probs[pairedArray[n_instances - 1].index] + 10e-6;
        points[n_instances - 1] = {.tp = tc[1][1], .fp = tc[1][0], .threshold = threshold};
    }

    free(probs);
    free(pairedArray);
    return points;
}
