#ifndef UTILH
#define UTILH

char* char_concat(char *c1, const char *c2);  //concatenate two strings with dynamic allocation

double randDouble(double min, double max);

int randInt(int min, int max);

double *getCurve(double *predictions, int n_instances, int n_classes, int classIndex);
double getROCArea(double *tcurve);


#endif
