#define _USE_MATH_DEFINES
#include <cmath>
extern "C" void f_c4d9efe9f75a09a5_dbg(double * ret, const double *params, const double *immed, const double eps) {
double s[3];
s[0] = immed[0];
s[1] = immed[1];
s[2] = params[0];
s[1] *= s[2];
s[0] += s[1];
*ret = s[0]; }
