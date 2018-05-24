#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com
template<typename T>
void reverseArray(T *inArr, Nd4jLong *inShapeBuffer, T *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse) {

}


///////////////////////////////////////////////////////////////////
template <typename T>
void reverseSequence(const NDArray<T>* input, const NDArray<T>* seqLengths, NDArray<T>* output, int seqDim, const int batchDim){

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void reverse(const NDArray<T>* input, NDArray<T>* output, const std::vector<int>* intArgs) {

}

template void reverseSequence<float>(const NDArray<float>* input, const NDArray<float>* seqLengths, NDArray<float>* output, int seqDim, const int batchDim);
template void reverseSequence<float16>(const NDArray<float16>* input, const NDArray<float16>* seqLengths, NDArray<float16>* output, int seqDim, const int batchDim);
template void reverseSequence<double>(const NDArray<double>* input, const NDArray<double>* seqLengths, NDArray<double>* output, int seqDim, const int batchDim);

template void reverseArray<float>(float *inArr, Nd4jLong *inShapeBuffer, float *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<float16>(float16 *inArr, Nd4jLong *inShapeBuffer, float16 *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<double>(double *inArr, Nd4jLong *inShapeBuffer, double *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);

template void reverse<float>(const NDArray<float>* input, NDArray<float>* output, const std::vector<int>* intArgs);
template void reverse<float16>(const NDArray<float16>* input, NDArray<float16>* output, const std::vector<int>* intArgs);
template void reverse<double>(const NDArray<double>* input, NDArray<double>* output, const std::vector<int>* intArgs);


}
}
}

