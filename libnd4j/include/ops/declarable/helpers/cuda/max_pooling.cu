//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/max_pooling.h>
#include <ops/declarable/generic/helpers/convolutions.h>

#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void maxPoolingFunctor(NDArray<T>* input, NDArray<T>* values, std::vector<int> const& params, NDArray<T>* indices) {

    }
    template void maxPoolingFunctor<float>(NDArray<float>* input, NDArray<float>* values, std::vector<int> const& params, NDArray<float>* indices);
    template void maxPoolingFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, std::vector<int> const& params, NDArray<float16>* indices);
    template void maxPoolingFunctor<double>(NDArray<double>* input, NDArray<double>* values, std::vector<int> const& params, NDArray<double>* indices);

}
}
}