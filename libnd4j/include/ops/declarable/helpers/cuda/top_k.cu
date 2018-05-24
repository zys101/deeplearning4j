#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>
#include <ops/declarable/headers/parity_ops.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
int topKFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indeces, int k, bool needSort) {

    return 58;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
int inTopKFunctor(NDArray<T>* input, NDArray<T>* target, NDArray<T>* result, int k) {

    return 58;
}

template int topKFunctor<float>(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indeces, int k, bool needSort);
template int topKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indeces, int k, bool needSort);
template int topKFunctor<double>(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indeces, int k, bool needSort);

template int inTopKFunctor<float>(NDArray<float>* input, NDArray<float>* target, NDArray<float>* result, int k);
template int inTopKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* target, NDArray<float16>* result, int k);
template int inTopKFunctor<double>(NDArray<double>* input, NDArray<double>* target, NDArray<double>* result, int k);

}
}
}