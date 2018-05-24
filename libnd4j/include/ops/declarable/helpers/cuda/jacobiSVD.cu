#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/hhColPivQR.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
JacobiSVD<T>::JacobiSVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnLeft(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnRight(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::isBlock2x2NotDiag(NDArray<T>& block, int p, int q, T& maxElem) {

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::createJacobiRotation(const T& x, const T& y, const T& z, NDArray<T>& rotation) {
  
  return true;  
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::svd2x2(const NDArray<T>& block, int p, int q, NDArray<T>& left, NDArray<T>& right) {
    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::evalData(const NDArray<T>& matrix) {

}

template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;

}
}
}

