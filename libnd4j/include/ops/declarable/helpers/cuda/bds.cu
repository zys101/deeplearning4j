#include <ops/declarable/helpers/bds.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
int bdsFunctor(NDArray<T>* x_shape, NDArray<T>* y_shape, NDArray<T>* output) {

    return 58;
}

template int bdsFunctor(NDArray<float>* x_shape, NDArray<float>* y_shape, NDArray<float>* output);
template int bdsFunctor(NDArray<float16>* x_shape, NDArray<float16>* y_shape, NDArray<float16>* output);
template int bdsFunctor(NDArray<double>* x_shape, NDArray<double>* y_shape, NDArray<double>* output);

}
}
}