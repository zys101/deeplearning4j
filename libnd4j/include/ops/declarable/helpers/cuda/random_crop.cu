#include <ops/declarable/helpers/random_crop.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* shape, NDArray<T>* output, int seed) {

    return 58;
}

template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<float>* input, NDArray<float>* shape, NDArray<float>* output,  int seed);
template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<float16>* input, NDArray<float16>* shape, NDArray<float16>* output, int seed);
template int randomCropFunctor(nd4j::random::RandomBuffer* rng, NDArray<double>* input, NDArray<double>* shape, NDArray<double>* output, int seed);

}
}
}