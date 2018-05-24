#include <ops/declarable/helpers/percentile.h>
#include "ResultSet.h"
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////    
template <typename T>
void percentile(const NDArray<T>& input, NDArray<T>& output, std::vector<int>& axises, const T q, const int interpolation) {
    
}

template void percentile(const NDArray<float>& input, NDArray<float>& output, std::vector<int>& axises, const float q, const int interpolation);
template void percentile(const NDArray<float16>& input, NDArray<float16>& output, std::vector<int>& axises, const float16 q, const int interpolation);
template void percentile(const NDArray<double>& input, NDArray<double>& output, std::vector<int>& axises, const double q, const int interpolation);

}
}
}