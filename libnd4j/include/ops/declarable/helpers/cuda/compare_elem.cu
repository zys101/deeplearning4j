#include <ops/declarable/helpers/compare_elem.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename T>
void compare_elem(NDArray<T> *input, bool isStrictlyIncreasing, bool& output) {

}

template void compare_elem<float>(NDArray<float> *A, bool isStrictlyIncreasing, bool& output);
template void compare_elem<float16>(NDArray<float16> *A, bool isStrictlyIncreasing, bool& output);
template void compare_elem<double>(NDArray<double> *A, bool isStrictlyIncreasing, bool& output);

}
}
}
