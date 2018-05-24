#include <ops/declarable/helpers/segment.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
// segment max
template <typename T>
void segmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {

}

///////////////////////////////////////////////////////////////////
// segmen min 
template <typename T>
void segmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {

}

///////////////////////////////////////////////////////////////////
// segmen mean
template <typename T>
void segmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
void segmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
void segmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
bool segmentIndicesValidate(NDArray<T>* indices, T& expected, T& output) {
    
    return true;
}

template bool segmentIndicesValidate(NDArray<float>* indices, float& expected, float& output);
template bool segmentIndicesValidate(NDArray<float16>* indices, float16& expected, float16& output);
template bool segmentIndicesValidate(NDArray<double>* indices, double& expected, double& output);

template void segmentMaxFunctor<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* output);
template void segmentMaxFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
template void segmentMaxFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

template void segmentMinFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
template void segmentMinFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
template void segmentMinFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

template void segmentMeanFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
template void segmentMeanFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
template void segmentMeanFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

template void segmentSumFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
template void segmentSumFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
template void segmentSumFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

template void segmentProdFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
template void segmentProdFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
template void segmentProdFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

}
}
}