#include <op_boilerplate.h>
#include <types/float16.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <helpers/BlasHelper.h>


namespace nd4j    {
namespace ops     {
namespace helpers {
        

//////////////////////////////////////////////////////////////////////////
template <typename T>
void _bgemm(std::vector<NDArray<T>*>& vA, std::vector<NDArray<T>*>& vB, std::vector<NDArray<T>*>& vC, NDArray<T>* alphas, NDArray<T>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC) {

}

template void _bgemm<float>(std::vector<NDArray<float>*>& vA, std::vector<NDArray<float>*>& vB, std::vector<NDArray<float>*>& vC, NDArray<float>* alphas, NDArray<float>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
template void _bgemm<double>(std::vector<NDArray<double>*>& vA, std::vector<NDArray<double>*>& vB, std::vector<NDArray<double>*>& vC, NDArray<double>* alphas, NDArray<double>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
template void _bgemm<float16>(std::vector<NDArray<float16>*>& vA, std::vector<NDArray<float16>*>& vB, std::vector<NDArray<float16>*>& vC, NDArray<float16>* alphas, NDArray<float16>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);

}
}
}