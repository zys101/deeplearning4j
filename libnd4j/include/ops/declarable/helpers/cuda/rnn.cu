#include<ops/declarable/helpers/rnn.h>
#include <helpers/BlasHelper.h>

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
void rnnCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h, NDArray<T>* hFinal) {

}



template void rnnCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* ht);
template void rnnCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* ht);
template void rnnCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* ht);

template void rnnTimeLoop<float>  (const std::vector<NDArray<float>*>&   inArrs, NDArray<float>*   h, NDArray<float>*   hFinal);
template void rnnTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h, NDArray<float16>* hFinal);
template void rnnTimeLoop<double> (const std::vector<NDArray<double>*>&  inArrs, NDArray<double>*  h, NDArray<double>*  hFinal);


}
}
}

