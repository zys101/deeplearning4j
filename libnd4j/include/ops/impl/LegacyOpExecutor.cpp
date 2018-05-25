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
    void LegacyOpExecutor<T>::execSummaryStatsScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras, bool biasCorrected) {
        T res = NativeOpExcutioner<T>::execSummaryStatsScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data(),  biasCorrected);
        z->putScalar(0, res);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execSummaryStatsOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras, bool biasCorrected) {
        NativeOpExcutioner<T>::execSummaryStats(opNum, x->getBuffer(), x->getShapeInfo(), extras.data(), z->getBuffer(), z->getShapeInfo(), axis.data(), static_cast<int>(axis.size()), biasCorrected);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduceScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {
        T res = NativeOpExcutioner<T>::execReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data());
        z->putScalar(0, res);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduceOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NativeOpExcutioner<T>::execReduce(opNum, x->buffer(), x->shapeInfo(), extras.data(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduce3ScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<T> &extras) {
        T scalar = NativeOpExcutioner<T>::execReduce3Scalar(opNum, x->buffer(), x->shapeInfo(), extras.data(), y->buffer(), y->shapeInfo());
        z->putScalar(0, scalar);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduce3Op(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        NativeOpExcutioner<T>::execReduce3(opNum, x->buffer(), x->shapeInfo(), extras.data(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()));
    }

    template <typename T>
    void LegacyOpExecutor<T>::execPairwiseOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<T> &extras) {
        NativeOpExcutioner<T>::execPairwiseTransform(opNum, x->getBuffer(), x->getShapeInfo(), y->getBuffer(), y->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), extras.data());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execIndexReduceScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {
        T res = NativeOpExcutioner<T>::execIndexReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data());
        z->putScalar(0, res);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execIndexReduceOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        NativeOpExcutioner<T>::execIndexReduce(opNum, x->buffer(), x->shapeInfo(), extras.data(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execBroadcastOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<int> &axis) {
        shape::TAD tad(x->shapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        if (x == z) {
            shape::TAD tadZ(z->shapeInfo(), axis.data(), static_cast<int>(axis.size()));
            tadZ.createTadOnlyShapeInfo();
            tadZ.createOffsets();

            NativeOpExcutioner<T>::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(),
                                                 z->buffer(), z->shapeInfo(), axis.data(),
                                                 static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets,
                                                 tadZ.tadOnlyShapeInfo, tadZ.tadOffsets);
        } else {
            NativeOpExcutioner<T>::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(),
                                                 z->buffer(), z->shapeInfo(), axis.data(),
                                                 static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets,
                                                 tad.tadOnlyShapeInfo, tad.tadOffsets);
        }
    }

    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}