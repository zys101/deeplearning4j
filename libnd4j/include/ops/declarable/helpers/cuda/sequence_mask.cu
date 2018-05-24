#include <ops/declarable/helpers/sequence_mask.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
void sequenceMask(NDArray<T>* input, NDArray<T>* output, int maxIndex) {

}

template void sequenceMask(NDArray<float>* input, NDArray<float>* output, int maxIndex);
template void sequenceMask(NDArray<float16>* input, NDArray<float16>* output, int maxIndex);
template void sequenceMask(NDArray<double>* input, NDArray<double>* output, int maxIndex);

}
}
}