#include <ops/declarable/helpers/weights.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

template <typename T>
void adjustWeights(NDArray<T>* input, NDArray<T>* weights, NDArray<T>* output, int minLength, int maxLength) {

}

template void adjustWeights(NDArray<float>* input, NDArray<float>* weights, NDArray<float>* output, int minLength, int maxLength);
template void adjustWeights(NDArray<float16>* input, NDArray<float16>* weights, NDArray<float16>* output, int minLength, int maxLength);
template void adjustWeights(NDArray<double>* input, NDArray<double>* weights, NDArray<double>* output, int minLength, int maxLength);

}
}
}