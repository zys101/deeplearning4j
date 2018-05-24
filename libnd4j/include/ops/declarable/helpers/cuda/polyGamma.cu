#include<ops/declarable/helpers/polyGamma.h>
#include<ops/declarable/helpers/zeta.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// calculate factorial
template <typename T>
static T getFactorial(const int n) {
	
	return T();
}

//////////////////////////////////////////////////////////////////////////
// implementation is based on serial representation written in terms of the Hurwitz zeta function as polygamma = (-1)^{n+1} * n! * zeta(n+1, x)
template <typename T>
static T polyGamma(const int n, const T x) {

	return T();	
}

//////////////////////////////////////////////////////////////////////////
// calculate polygamma function for arrays
template <typename T>
NDArray<T> polyGamma(const NDArray<T>& n, const NDArray<T>& x) {

	return NDArray<T>();
}


template float   polyGamma<float>  (const int n, const float   x);
template float16 polyGamma<float16>(const int n, const float16 x);
template double  polyGamma<double> (const int n, const double  x);

template NDArray<float>   polyGamma<float>  (const NDArray<float>&   n, const NDArray<float>&   x);
template NDArray<float16> polyGamma<float16>(const NDArray<float16>& n, const NDArray<float16>& x);
template NDArray<double>  polyGamma<double> (const NDArray<double>&  n, const NDArray<double>&  x);


}
}
}

