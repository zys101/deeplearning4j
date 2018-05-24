#include <ops/declarable/helpers/lrn.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
int lrnFunctor(NDArray<T>* input, NDArray<T>* output, int depth, T bias, T alpha, T beta) {

    return 58;
}

///////////////////////////////////////////////////////////////////
template <typename T>
int lrnFunctorEx(NDArray<T>* input, NDArray<T>* output, NDArray<T>* unitScale, NDArray<T>* scale, int depth, T bias, T alpha, T beta) {
    
    return 58;
}

    template int lrnFunctor(NDArray<float>* input, NDArray<float>* output, int depth, float bias, float alpha, float beta);
    template int lrnFunctor(NDArray<float16>* input, NDArray<float16>* output, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctor(NDArray<double>* input, NDArray<double>* output, int depth, double bias, double alpha, double beta);
    template int lrnFunctorEx(NDArray<float>* input, NDArray<float>* output, NDArray<float>* unitScale, NDArray<float>* scale, int depth, float bias, float alpha, float beta);
    template int lrnFunctorEx(NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* unitScale, NDArray<float16>* scale, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctorEx(NDArray<double>* input, NDArray<double>* output, NDArray<double>* unitScale, NDArray<double>* scale, int depth, double bias, double alpha, double beta);
}
}
}