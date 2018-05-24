#include <ops/declarable/helpers/svd.h>
#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(const NDArray<T>& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV ) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(const NDArray<T>& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV, const char t) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation1(int col1, int shift, int ind, int size) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation2(int col1U , int col1M, int row1W, int col1W, int ind1, int ind2, int size) {

}

//////////////////////////////////////////////////////////////////////////
// has effect on block from (col1+shift, col1+shift) to (col2+shift, col2+shift) inclusively 
template <typename T>
void SVD<T>::deflation(int col1, int col2, int ind, int row1W, int col1W, int shift) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
T SVD<T>::secularEq(const T diff, const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, const NDArray<T>& diagShifted, const T shift) {
    return T();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVals(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, NDArray<T>& singVals, NDArray<T>& shifts, NDArray<T>& mus) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T> 
void SVD<T>::perturb(const NDArray<T>& col0, const NDArray<T>& diag, const NDArray<T>& permut, const NDArray<T>& singVals,  const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& zhat) {    

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVecs(const NDArray<T>& zhat, const NDArray<T>& diag, const NDArray<T>& perm, const NDArray<T>& singVals,
                             const NDArray<T>& shifts, const NDArray<T>& mus, NDArray<T>& U, NDArray<T>& V) {
  
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcBlockSVD(int col1, int size, NDArray<T>& U, NDArray<T>& singVals, NDArray<T>& V) {  
    
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void SVD<T>::DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void SVD<T>::exchangeUV(const HHsequence<T>& hhU, const HHsequence<T>& hhV, const NDArray<T> U, const NDArray<T> V) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::evalData(const NDArray<T>& matrix) {

}


template class ND4J_EXPORT SVD<float>;
template class ND4J_EXPORT SVD<float16>;
template class ND4J_EXPORT SVD<double>;

//////////////////////////////////////////////////////////////////////////
// svd operation, this function is not method of SVD class, it is standalone function
template <typename T>
void svd(const NDArray<T>* x, const std::vector<NDArray<T>*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum) {

}

template void svd<float>(const NDArray<float>* x, const std::vector<NDArray<float>*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum);
template void svd<float16>(const NDArray<float16>* x, const std::vector<NDArray<float16>*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum);
template void svd<double>(const NDArray<double>* x, const std::vector<NDArray<double>*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum);





}
}
}

