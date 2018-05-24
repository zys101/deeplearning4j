#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T> 
void swapRows(NDArray<T>* matrix, int theFirst, int theSecond) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
void invertLowerMatrix(NDArray<T>* inputMatrix, NDArray<T>* invertedMatrix) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
void invertUpperMatrix(NDArray<T>* inputMatrix, NDArray<T>* invertedMatrix) {

}

///////////////////////////////////////////////////////////////////
template <typename T>
T lup(NDArray<T>* input, NDArray<T>* compound, NDArray<T>* permutation) {

    return T();
}

///////////////////////////////////////////////////////////////////
template <typename T>
int determinant(NDArray<T>* input, NDArray<T>* output) {

    return 58;
}

///////////////////////////////////////////////////////////////////
template <typename T>
int inverse(NDArray<T>* input, NDArray<T>* output) {
    
    return 58;
}

    template void swapRows(NDArray<float>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<float16>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<double>* matrix, int theFirst, int theSecond);

    template void invertLowerMatrix(NDArray<float>* inputMatrix, NDArray<float>* invertedMatrix);
    template void invertLowerMatrix(NDArray<float16>* inputMatrix, NDArray<float16>* invertedMatrix);
    template void invertLowerMatrix(NDArray<double>* inputMatrix, NDArray<double>* invertedMatrix);

    template void invertUpperMatrix(NDArray<float>* inputMatrix, NDArray<float>* invertedMatrix);
    template void invertUpperMatrix(NDArray<float16>* inputMatrix, NDArray<float16>* invertedMatrix);
    template void invertUpperMatrix(NDArray<double>* inputMatrix, NDArray<double>* invertedMatrix);

    template float lup(NDArray<float>* input, NDArray<float>* output, NDArray<float>* permutation);
    template float16 lup(NDArray<float16>* input, NDArray<float16>* compound, NDArray<float16>* permutation);
    template double lup(NDArray<double>* input, NDArray<double>* compound, NDArray<double>* permutation);

    template int determinant(NDArray<float>* input, NDArray<float>* output);
    template int determinant(NDArray<float16>* input, NDArray<float16>* output);
    template int determinant(NDArray<double>* input, NDArray<double>* output);

    template int inverse(NDArray<float>* input, NDArray<float>* output);
    template int inverse(NDArray<float16>* input, NDArray<float16>* output);
    template int inverse(NDArray<double>* input, NDArray<double>* output);
}
}
}