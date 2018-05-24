
#include <ops/declarable/helpers/adjust_hue.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE void _adjust_hue_single(NDArray<T> *array, NDArray<T> *output, T delta, bool isNHWC) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void _adjust_hue(NDArray<T> *array, NDArray<T> *output, T delta, bool isNHWC) {

}

template void _adjust_hue<float>(NDArray<float> *array, NDArray<float> *output, float delta, bool isNHWC);
template void _adjust_hue<float16>(NDArray<float16> *array, NDArray<float16> *output, float16 delta, bool isNHWC);
template void _adjust_hue<double>(NDArray<double> *array, NDArray<double> *output, double delta, bool isNHWC);
}
}
}