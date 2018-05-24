#include <ops/declarable/helpers/confusion.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void confusionFunctor(NDArray<T>* labels, NDArray<T>* predictions, NDArray<T>* weights, NDArray<T>* output) {

}

template void confusionFunctor(NDArray<float>* labels, NDArray<float>* predictions, NDArray<float>* weights, NDArray<float>* output);
template void confusionFunctor(NDArray<float16>* labels, NDArray<float16>* predictions, NDArray<float16>* weights, NDArray<float16>* output);
template void confusionFunctor(NDArray<double>* labels, NDArray<double>* predictions, NDArray<double>* weights, NDArray<double>* output);

}
}
}