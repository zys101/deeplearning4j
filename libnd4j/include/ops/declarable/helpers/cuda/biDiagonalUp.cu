#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
BiDiagonalUp<T>::BiDiagonalUp(const NDArray<T>& matrix): _HHmatrix(NDArray<T>(matrix.ordering(), {matrix.sizeAt(0), matrix.sizeAt(1)}, matrix.getWorkspace())),  
                                                         _HHbidiag(NDArray<T>(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.getWorkspace())) {
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void BiDiagonalUp<T>::evalData() {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence<T> BiDiagonalUp<T>::makeHHsequence(const char type) const {
	
	return HHsequence<T>(NDArray<T>(), NDArray<T>(), 'u');	
}


template class ND4J_EXPORT BiDiagonalUp<float>;
template class ND4J_EXPORT BiDiagonalUp<float16>;
template class ND4J_EXPORT BiDiagonalUp<double>;

}
}
}