#include <ops/declarable/helpers/householder.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> Householder<T>::evalHHmatrix(const NDArray<T>& x) {

	return NDArray<T>();	
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(const NDArray<T>& x, NDArray<T>& tail, T& coeff, T& normX) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(const NDArray<T>& x, T& coeff, T& normX) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {
	
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulRight(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {

}

      
template class ND4J_EXPORT Householder<float>;
template class ND4J_EXPORT Householder<float16>;
template class ND4J_EXPORT Householder<double>;


}
}
}
