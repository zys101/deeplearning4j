#include<ops/declarable/helpers/activations.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
void softMaxForVector(const NDArray<T>& input, NDArray<T>& output) {

}


///////////////////////////////////////////////////////////////////
template <typename T>
void logSoftMaxForVector(const NDArray<T>& input, NDArray<T>& output) {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void softmax(const NDArray<T>& input, NDArray<T>& output, const int dimension) {

}


template void softMaxForVector<float>  (const NDArray<float  >& input, NDArray<float  >& output);
template void softMaxForVector<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void softMaxForVector<double> (const NDArray<double >& input, NDArray<double >& output);

template void logSoftMaxForVector<float>  (const NDArray<float  >& input, NDArray<float  >& output);
template void logSoftMaxForVector<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void logSoftMaxForVector<double> (const NDArray<double >& input, NDArray<double >& output);

template void softmax<float>(const NDArray<float>& input, NDArray<float>& output, const int dimension);
template void softmax<float16>(const NDArray<float16>& input, NDArray<float16>& output, const int dimension);
template void softmax<double>(const NDArray<double>& input, NDArray<double>& output, const int dimension);


}
}
}

