#define _USE_MATH_DEFINES
#include <cmath>
extern "C" void f_c4d9ef9f3290ea80_dbg(double * ret, const double *params, const double *immed, const double eps) {
double s[3];
s[0] = immed[0];
s[1] = params[0];
s[2] = immed[1];
s[1] *= s[2];
s[0] += s[1];
*ret = s[0]; }
