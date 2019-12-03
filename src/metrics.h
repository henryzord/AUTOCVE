#ifndef METRICSH
#define METRICSH

#include <Python.h>
#include "python_interface.h"

struct pair {
    double value;
    int index;
} typedef pair_t;

struct roc_point {
    double tp;
    double fp;
    double threshold;
} typedef roc_point_t;

int compare(const void *a, const void *b);
roc_point_t *getCurve(PyObject *y_score, int *y_true_int, int classIndex, int *count_vec);
double getROCArea(roc_point_t *points, int n_instances);

#endif // METRICSH