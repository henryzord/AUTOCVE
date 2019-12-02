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

int compare(const void *a, const void *b) {
  return ((pair_t*)a)->value >= ((pair_t*)b)->value;
}

roc_point_t *getCurve(PyObject *y_pred, int classIndex) {
    if(y_pred == NULL) {
        return NULL;
    }

    npy_intp *y_pred_dims = PyArray_DIMS((PyArrayObject*)y_pred);
    int n_instances = (int)y_pred_dims[0], n_classes = (int)y_pred_dims[1];

    if ((n_instances == 0) || (n_classes <= classIndex)) {
      return NULL;
    }

    npy_intp y_pred_itemsize = PyArray_ITEMSIZE((PyArrayObject*)y_pred);

    char *y_pred_ptr = PyArray_BYTES((PyArrayObject*)y_pred);
    int *pred = (int*)malloc(sizeof(int) * n_instances);

    pair_t *pairedArray = (pair_t*)malloc(sizeof(pair_t) * n_instances);

    // Weka code starts below

    int totPos = 0, totNeg = 0;

    // Get distribution of positive/negatives
    double prob, max_prob;
    int max_index;
    for (int i = 0; i < n_instances; i++) {
        max_prob = -1;
        max_index = -1;

        for(int j = 0; j < n_classes; j++) {
            prob = PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)y_pred, y_pred_ptr));
            y_pred_ptr += y_pred_itemsize;

            if(prob > max_prob) {
                max_prob = prob;
                max_index = j;
            }
            if(j == classIndex) {
                pairedArray[i] = {.value = prob, .index = i};
            }
        }

        pred[i] = max_index;
        if(max_index == classIndex) {
            totPos += 1;
        } else {
            totNeg += 1;
        }
    }

    qsort(pairedArray, n_instances, sizeof(pair_t), compare);

    double threshold = 0;
    int cumulativePos = 0, cumulativeNeg = 0;
    int fp = totNeg, fn = 0, tp = totPos, tn = 0;

    roc_point_t *points = (roc_point_t*)malloc(sizeof(roc_point_t) * (n_instances + 1));

    for(int i = 0; i < n_instances; i++) {
        if((i == 0) || (pairedArray[i].value > threshold)) {
            tp = tp - cumulativePos;  // true positive
            fn = fn + cumulativePos;  // false negative
            fp = fp - cumulativeNeg;  // false positive
            tn = tn + cumulativeNeg;  // true negative
            threshold = pairedArray[i].value;
            points[i] = {.tp = tp, .fp = fp, .threshold = threshold};
            cumulativeNeg = 0;
            cumulativePos = 0;
            if(i == (n_instances - 1)) {
                break;
            }
        }

        if(pred[pairedArray[i].index] == classIndex) {
            cumulativePos += 1;
        } else {
            cumulativeNeg += 1;
        }
    }

    // make sure a zero point gets into the curve
    if((fn != totPos) || (tn != totNeg)) {
        tp = 0;
        fp = 0;
        tn = totNeg;
        fn = totPos;
        threshold = pairedArray[n_instances - 1].value + 10e-6;
        points[n_instances] = {.tp = tp, .fp = fp, .threshold = threshold};
    } else {
        points[n_instances] = {.tp = -1, .fp = -1, .threshold = -1};
    }

    free(pred);
    free(pairedArray);
    return points;
}

double getROCArea(roc_point_t *points, int n_instances) {
    int n;
    if(points[n_instances].threshold == -1) {
        n = n_instances;
    } else {
        n = n_instances + 1;
    }
    if(n == 0) {
        return NAN;
    }

    double area = 0.0, cumNeg = 0.0;
    double totalPos = points[0].tp;
    double totalNeg = points[0].fp;
    for(int i = 0; i < n; i++) {
        double cip, cin;
        if(i < (n - 1)) {
            cip = points[i].tp - points[i + 1].tp;
            cin = points[i].fp - points[i + 1].fp;
        }  else {
            cip = points[n - 1].tp;
            cin = points[n - 1].fp;
        }
        area += cip * (cumNeg + (0.5 * cin));
        cumNeg += cin;
    }
    area /= (totalNeg * totalPos);

    return area;
}