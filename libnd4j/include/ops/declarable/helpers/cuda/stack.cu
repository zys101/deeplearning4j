#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
void stack(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& outArr, const int dim) {

}


template void stack<float>  (const std::vector<NDArray<float  >*>& inArrs, NDArray<float  >& outArr, const int dim);
template void stack<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& outArr, const int dim);
template void stack<double> (const std::vector<NDArray<double >*>& inArrs, NDArray<double >& outArr, const int dim);


}
}
}

