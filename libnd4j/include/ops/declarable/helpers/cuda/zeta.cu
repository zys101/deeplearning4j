#include<ops/declarable/helpers/zeta.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

const int maxIter = 1000000;							// max number of loop iterations 
const double machep =  1.11022302462515654042e-16;		

// expansion coefficients for Euler-Maclaurin summation formula (2k)! / B2k, where B2k are Bernoulli numbers
const double coeff[] = { 12.0,-720.0,30240.0,-1209600.0,47900160.0,-1.8924375803183791606e9,7.47242496e10,-2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15, 1.8152105401943546773e17, -7.1661652561756670113e18};


//////////////////////////////////////////////////////////////////////////
// slow implementation
template <typename T>
static T zetaSlow(const T x, const T q) {

	return T();
}


//////////////////////////////////////////////////////////////////////////
// fast implementation, it is based on Euler-Maclaurin summation formula
template <typename T>
T zeta(const T x, const T q) {
	
	return T();
}


//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function for arrays
template <typename T>
NDArray<T> zeta(const NDArray<T>& x, const NDArray<T>& q) {

	return NDArray<T>();
}


template float   zeta<float>  (const float   x, const float   q);
template float16 zeta<float16>(const float16 x, const float16 q);
template double  zeta<double> (const double  x, const double  q);

template NDArray<float>   zeta<float>  (const NDArray<float>&   x, const NDArray<float>&   q);
template NDArray<float16> zeta<float16>(const NDArray<float16>& x, const NDArray<float16>& q);
template NDArray<double>  zeta<double> (const NDArray<double>&  x, const NDArray<double>&  q);


}
}
}

