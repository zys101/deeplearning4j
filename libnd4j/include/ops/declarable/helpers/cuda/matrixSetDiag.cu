#include "ResultSet.h"
#include "NDArrayFactory.h"
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
void matrixSetDiag(const NDArray<T>* input, const NDArray<T>* diagonal, NDArray<T>* output) {

}



template void matrixSetDiag<float>(const NDArray<float>* input, const NDArray<float>* diagonal, NDArray<float>* output);
template void matrixSetDiag<float16>(const NDArray<float16>* input, const NDArray<float16>* diagonal, NDArray<float16>* output);
template void matrixSetDiag<double>(const NDArray<double>* input, const NDArray<double>* diagonal, NDArray<double>* output);


}
}
}