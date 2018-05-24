#include<ops/declarable/helpers/meshgrid.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>
#include <numeric>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
void meshgrid(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const bool swapFirst2Dims) {
    
}


template void meshgrid<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const bool swapFirst2Dims);
template void meshgrid<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const bool swapFirst2Dims);
template void meshgrid<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const bool swapFirst2Dims);

}
}
}

