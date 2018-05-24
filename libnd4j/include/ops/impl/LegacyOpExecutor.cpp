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

    template <typename T>
    void LegacyOpExecutor<T>::execTransformOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {
        NativeOpExcutioner<T>::execTransform(opNum, x->buffer(), x->shapeInfo(), z->getBuffer(), z->getShapeInfo(), extras.data(), nullptr, nullptr);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execSummaryStatsScalar(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras, bool biasCorrected) {
        T res = NativeOpExcutioner<T>::execSummaryStatsScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data(),  biasCorrected);
        z->putScalar(0, res);
    }

    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}