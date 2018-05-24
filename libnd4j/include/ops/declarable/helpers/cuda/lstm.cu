#include<ops/declarable/helpers/lstm.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return NDArray<T>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void clipping(NDArray<T>* arr, T limit) {    
    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void lstmCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void lstmTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params) {
    
}


template void clipping<float>(NDArray<float>* arr, float limit);
template void clipping<float16>(NDArray<float16>* arr, float16 limit);
template void clipping<double>(NDArray<double>* arr, double limit);

template void lstmCell<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const std::vector<float>& params);
template void lstmCell<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const std::vector<float16>& params);
template void lstmCell<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const std::vector<double>& params);

template void lstmTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const std::vector<float>& params);
template void lstmTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const std::vector<float16>& params);
template void lstmTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const std::vector<double>& params);


}
}
}

