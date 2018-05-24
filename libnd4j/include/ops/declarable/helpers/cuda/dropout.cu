#include <ops/declarable/helpers/dropout.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* output, NDArray<T>* reduceShape, int seed, T probValue) {

    return 58;
}

template int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray<float>* input, NDArray<float>* output, NDArray<float>* reduceShape, int seed, float probValue);
template int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* reduceShape, int seed, float16 probValue);
template int dropOutFunctor(nd4j::random::RandomBuffer* rng, NDArray<double>* input, NDArray<double>* output, NDArray<double>* reduceShape, int seed, double probValue);

}
}
}