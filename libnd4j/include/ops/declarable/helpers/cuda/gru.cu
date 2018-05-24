#include<ops/declarable/helpers/gru.h>

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
void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

}



template void gruCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

template void gruTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

}
}
}

