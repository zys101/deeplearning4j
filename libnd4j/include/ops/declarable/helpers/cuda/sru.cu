#include<ops/declarable/helpers/sru.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return NDArray<T>();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return NDArray<T>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void sruCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void sruTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {
    
}


template void sruCell<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs);
template void sruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs);
template void sruCell<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs);

template void sruTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs);
template void sruTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs);
template void sruTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs);


}
}
}