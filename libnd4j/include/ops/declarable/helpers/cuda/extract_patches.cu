#include <ops/declarable/helpers/axis.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
void extractPatches(NDArray<T>* images, NDArray<T>* output, 
        int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){

}

template void extractPatches(NDArray<float>* input, NDArray<float>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);
template void extractPatches(NDArray<float16>* input, NDArray<float16>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);
template void extractPatches(NDArray<double>* input, NDArray<double>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);

}
}
}