#ifndef UTILH
#define UTILH

char* char_concat(char *c1, const char *c2);  //concatenate two strings with dynamic allocation

double randDouble(double min, double max);

int randInt(int min, int max);

double *getCurve(double *predictions, int n_instances, int n_classes, int classIndex);
double getROCArea(double *tcurve);
void get_min_median_max_double(double *min_fit, double *median, double *max_fit, int *count_valid, int *n_discarded, double *pool, int size, int invalid_val);
void get_min_median_max_int(int *min_fit, int *median, int *max_fit, int *count_valid, int *n_discarded, int *pool, int size, int invalid_val);


#endif
