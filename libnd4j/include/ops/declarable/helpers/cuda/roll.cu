#include <ops/declarable/helpers/roll.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
void rollFunctorLinear(NDArray<T>* input, NDArray<T>* output, int shift, bool inplace){

}

///////////////////////////////////////////////////////////////////
template <typename T>
void rollFunctorFull(NDArray<T>* input, NDArray<T>* output, int shift, std::vector<int> const& axes, bool inplace) {
                

}

template void rollFunctorLinear(NDArray<float>*   input, NDArray<float>*   output, int shift, bool inplace);
template void rollFunctorLinear(NDArray<float16>* input, NDArray<float16>* output, int shift, bool inplace);
template void rollFunctorLinear(NDArray<double>*  input, NDArray<double>*  output, int shift, bool inplace);

template void rollFunctorFull(NDArray<float>*   input, NDArray<float>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
template void rollFunctorFull(NDArray<float16>* input, NDArray<float16>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
template void rollFunctorFull(NDArray<double>*  input, NDArray<double>* axisVector, int shift, std::vector<int> const& axes, bool inplace);

}
}
}