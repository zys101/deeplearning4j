#include <ops/declarable/helpers/hhColPivQR.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHcolPivQR<T>::HHcolPivQR(const NDArray<T>& matrix) {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHcolPivQR<T>::evalData() {

}




template class ND4J_EXPORT HHcolPivQR<float>;
template class ND4J_EXPORT HHcolPivQR<float16>;
template class ND4J_EXPORT HHcolPivQR<double>;







}
}
}

