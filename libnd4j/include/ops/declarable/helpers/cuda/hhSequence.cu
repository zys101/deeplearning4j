#include <ops/declarable/helpers/hhSequence.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence<T>::HHsequence(const NDArray<T>& vectors, const NDArray<T>& coeffs, const char type): _vectors(vectors), _coeffs(coeffs) {
	
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence<T>::mulLeft(NDArray<T>& matrix) const {    		

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> HHsequence<T>::getTail(const int idx) const {

    return NDArray<T>();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence<T>::applyTo(NDArray<T>& dest) const{

}


template class ND4J_EXPORT HHsequence<float>;
template class ND4J_EXPORT HHsequence<float16>;
template class ND4J_EXPORT HHsequence<double>;







}
}
}
