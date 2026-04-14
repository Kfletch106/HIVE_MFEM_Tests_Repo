#define _USE_MATH_DEFINES
#include <cmath>
extern "C" void f_39df4cda959f6b95_dbg(double * ret, const double *params, const double *immed, const double eps) {
double s[4];
s[0] = immed[0];
s[1] = params[1];
s[2] = params[3];
s[3] = params[2];
s[2] = s[3] - s[2];
s[1] *= s[2];
s[0] += s[1];
s[1] = params[0];
s[0] = s[1] / s[0];
*ret = s[0]; }
