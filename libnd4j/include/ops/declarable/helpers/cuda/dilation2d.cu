#include <ops/declarable/helpers/dilation2d.h>
#include <array/DataTypeUtils.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void _dilation2d(NDArray<T> *input, NDArray<T> *weights, NDArray<T> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {

}


template void _dilation2d<float>(NDArray<float> *input, NDArray<float> *weights, NDArray<float> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);
template void _dilation2d<float16>(NDArray<float16> *input, NDArray<float16> *weights, NDArray<float16> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);
template void _dilation2d<double>(NDArray<double> *input, NDArray<double> *weights, NDArray<double> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);

}
}
}