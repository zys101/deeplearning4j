#include <ops/declarable/helpers/matmul.h>

namespace nd4j     {
namespace ops      {
namespace helpers  {

///////////////////////////////////////////////////////////////////    
template <typename T>
void _matmul(NDArray<T> *vA, NDArray<T> *vB, NDArray<T> *vC, int transA, int transB, T alpha, T beta) {

}


template void _matmul<float>(NDArray<float> *A, NDArray<float> *B, NDArray<float> *C, int transA, int transB, float alpha, float beta);
template void _matmul<float16>(NDArray<float16> *A, NDArray<float16> *B, NDArray<float16> *C, int transA, int transB, float16 alpha, float16 beta);
template void _matmul<double>(NDArray<double> *A, NDArray<double> *B, NDArray<double> *C, int transA, int transB, double alpha, double beta);


}
}
}
