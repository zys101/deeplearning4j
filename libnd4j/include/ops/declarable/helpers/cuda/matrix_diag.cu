#include "ResultSet.h"
#include "NDArrayFactory.h"
#include <ops/declarable/helpers/matrix_diag.h>

namespace nd4j    { 
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
int matrixDiag(const NDArray<T>* input, NDArray<T>* output) {

    return 58;
}


template int matrixDiag<float>(const NDArray<float>* input, NDArray<float>* output);
template int matrixDiag<float16>(const NDArray<float16>* input, NDArray<float16>* output);
template int matrixDiag<double>(const NDArray<double>* input, NDArray<double>* output);


}
}
}