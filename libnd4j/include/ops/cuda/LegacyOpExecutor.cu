//
//
//

#include <ops/LegacyOpExecutor.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <loops/scalar.h>


namespace nd4j {

    template <typename T>
    void LegacyOpExecutor<T>::execScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, T scalar, std::vector<T> &extras) {
        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};

        functions::scalar::ScalarTransform<T>::executeCudaShaped(launchDims, extraPtrs, opNum, x->specialBuffer(), x->specialShapeInfo(), z->specialBuffer(),  z->specialShapeInfo(), scalar, extras.data());
    }


    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}