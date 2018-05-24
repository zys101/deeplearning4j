#include <ops/declarable/helpers/unique.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
int uniqueCount(NDArray<T>* input) {
    
    return 58;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
int uniqueFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indices, NDArray<T>* counts) { 
    
    return 58;
}

template int uniqueCount(NDArray<float>* input);
template int uniqueCount(NDArray<float16>* input);
template int uniqueCount(NDArray<double>* input);

template int uniqueFunctor(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indices, NDArray<float>* counts);
template int uniqueFunctor(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indices, NDArray<float16>* counts);
template int uniqueFunctor(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indices, NDArray<double>* counts);

}
}
}