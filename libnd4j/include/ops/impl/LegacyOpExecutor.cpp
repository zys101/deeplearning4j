//
// Created by raver on 5/23/2018.
//

#include <ops/LegacyOpExecutor.h>
#include <NativeOpExcutioner.h>

namespace nd4j {

    template <typename T>
    void LegacyOpExecutor<T>::execScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, T scalar, std::vector<T> &extras) {
        NativeOpExcutioner<T>::execScalar(opNum, x->getBuffer(), x->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), scalar, extras.data());
    }


    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}