#include <ops/declarable/helpers/dynamic.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void dynamicPartitionFunctor(NDArray<T>* input, NDArray<T>* indices, std::vector<NDArray<T>*>& outputList) {
                
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
int dynamicStitchFunctor(std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& indices, NDArray<T>* output){
    
    return 58;
}

template void dynamicPartitionFunctor(NDArray<float>* input, NDArray<float>* indices, std::vector<NDArray<float>*>& outputList);
template void dynamicPartitionFunctor(NDArray<float16>* input, NDArray<float16>* indices, std::vector<NDArray<float16>*>& outputList);
template void dynamicPartitionFunctor(NDArray<double>* input, NDArray<double>* indices, std::vector<NDArray<double>*>& outputList);

template int dynamicStitchFunctor(std::vector<NDArray<float>*>& inputs, std::vector<NDArray<float>*>& indices, NDArray<float>* output);
template int dynamicStitchFunctor(std::vector<NDArray<float16>*>& inputs, std::vector<NDArray<float16>*>& indices, NDArray<float16>* output);
template int dynamicStitchFunctor(std::vector<NDArray<double>*>& inputs, std::vector<NDArray<double>*>& indices, NDArray<double>* output);

}
}
}

