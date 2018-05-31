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
#include <loops/transform.h>
#include <loops/reduce.h>
#include <loops/reduce3.h>
#include <loops/indexreduce.h>
#include <loops/broadcasting.h>
#include <loops/pairwise_transform.h>
#include <helpers/TadMigrationHelper.h>
#include <helpers/VectorMigrationHelper.h>


namespace nd4j {

    template <typename T>
    void LegacyOpExecutor<T>::execScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, T scalar, std::vector<T> &extras) {
        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};

        functions::scalar::ScalarTransform<T>::executeCudaShaped(launchDims, extraPtrs, opNum, x->specialBuffer(), x->specialShapeInfo(), z->specialBuffer(),  z->specialShapeInfo(), scalar, extras.data());
        cudaStreamSynchronize(*ctx.stream());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execSummaryStatsScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras, bool biasCorrected) {
//        T res = NativeOpExcutioner<T>::execSummaryStatsScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data(),  biasCorrected);
//        z->putScalar(0, res);

        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};
//dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *extraParams, bool biasCorrected
        (*z)(0) = functions::summarystats::SummaryStatsReduce<T>::execSummaryStatsReduceScalar(launchDims, extraPtrs, opNum, x->specialBuffer(), x->specialShapeInfo(), extras.data(), biasCorrected);
        cudaStreamSynchronize(*ctx.stream());

    }

    template <typename T>
    void LegacyOpExecutor<T>::execSummaryStatsOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras, bool biasCorrected) {
        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};
//dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *extraParams, bool biasCorrected
        functions::summarystats::SummaryStatsReduce<T>::execSummaryStatsReduce(launchDims, extraPtrs, opNum, x->specialBuffer(), x->specialShapeInfo(), extras.data(), z->specialBuffer(),  z->specialShapeInfo(), biasCorrected);

//        _CUDA_H void SummaryStatsReduce<float>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, Nd4jLong *xShapeInfo, float *extraParams, float *result, Nd4jLong *resultShapeInfo,bool biasCorrected) {
          
//        NativeOpExcutioner<T>::execSummaryStats(opNum, x->getBuffer(), x->getShapeInfo(), extras.data(), z->getBuffer(), z->getShapeInfo(), axis.data(), static_cast<int>(axis.size()), biasCorrected);
        cudaStreamSynchronize(*ctx.stream());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execTransformOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {
        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};

        std::vector<int> axis(x->rankOf());
        for (int i = 0; i < axis.size(); i++)
            axis[i] = i;

        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
//        tad.createTadOnlyShapeInfo();
//        tad.createOffsets();
        TadMigrationHelper helper(tad);

//        NativeOpExcutioner<T>::execTransform(opNum, x->buffer(), x->shapeInfo(), z->getBuffer(), z->getShapeInfo(), extras.data(), nullptr, nullptr);
//	executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, Nd4jLong *xShape, int xRank, T *extraParams, T *z, Nd4jLong *zShape, int zRank, int *allocationPointer, T *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

        functions::transform::Transform<T>::executeTransformShaped(launchDims, ctx.stream(), opNum, x->specialBuffer(), x->specialShapeInfo(), x->rankOf(), extras.data(), z->specialBuffer(),  z->specialShapeInfo(), z->rankOf(), reinterpret_cast<int *>(ctx.allocationBuffer()), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo(), helper.tadOffsets());
        cudaStreamSynchronize(*ctx.stream());

    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduceScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {
//        T res = NativeOpExcutioner<T>::execReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), extras.data());
//        z->putScalar(0, res);
//        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        std::vector<int> axis(x->rankOf());
        for (int i = 0; i < axis.size(); i++)
            axis[i] = i;

        dim3 launchDims = {128, 1024, 2048};
//execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, float *x, Nd4jLong *xShapeInfo, float *extraParams, float *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        //tad.createTadOnlyShapeInfo();
        //tad.createOffsets();

        TadMigrationHelper helper(tad);
        VectorMigrationHelper<int> _axis(axis);
        VectorMigrationHelper<T> _extras(extras);

        functions::reduce::ReduceFunction<T>::execReduceScalar(launchDims, ctx.stream(), opNum, x->specialBuffer(), x->specialShapeInfo(), _extras.data(), z->specialBuffer(),  z->specialShapeInfo(), _axis.data(), axis.size(), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo());
        cudaStreamSynchronize(*ctx.stream());

    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduceOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
//        tad.createTadOnlyShapeInfo();
//        tad.createOffsets();

        TadMigrationHelper helper(tad);

        //NativeOpExcutioner<T>::execReduce(opNum, x->buffer(), x->shapeInfo(), extras.data(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets);

        dim3 launchDims = {128, 1024, 8192};

        VectorMigrationHelper<int> _axis(axis);
        VectorMigrationHelper<T> _extras(extras);

        functions::reduce::ReduceFunction<T>::execReduceXD(launchDims, ctx.stream(), opNum, x->rankOf(), x->specialBuffer(), x->specialShapeInfo(), _extras.data(), z->specialBuffer(), z->specialShapeInfo(), _axis.data(), axis.size(), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo(), helper.tadOffsets());

        cudaStreamSynchronize(*ctx.stream());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execBroadcastOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<int> &axis) {

        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        shape::TAD tadZ(z->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));

        TadMigrationHelper helper(tad);
        TadMigrationHelper helperZ(tadZ);

        //NativeOpExcutioner<T>::execReduce(opNum, x->buffer(), x->shapeInfo(), extras.data(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets);

        dim3 launchDims = {128, 1024, 8192};

        VectorMigrationHelper<int> _axis(axis);
//        VectorMigrationHelper<T> _extras(extras);


//  executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, Nd4jLong *xShapeInfo, T *y, Nd4jLong *yShapeInfo, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ);
        functions::broadcast::Broadcast<T>::executeBroadcast(launchDims, ctx.stream(), opNum, x->specialBuffer(), x->specialShapeInfo(), y->specialBuffer(), y->specialShapeInfo(), z->specialBuffer(), z->specialShapeInfo(), _axis.data(), axis.size(), helper.tadShapeInfo(), helper.tadOffsets(), helperZ.tadShapeInfo(), helperZ.tadOffsets());
        cudaStreamSynchronize(*ctx.stream());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execIndexReduceScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<T> &extras) {

//    executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
        std::vector<int> axis(x->rankOf());
        for (int i = 0; i < axis.size(); i++)
            axis[i] = i;

        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        TadMigrationHelper helper(tad);
        dim3 launchDims = {128, 1024, 8192};

        VectorMigrationHelper<int> _axis(axis);
        VectorMigrationHelper<T> _extras(extras);
        functions::indexreduce::IndexReduce<T>::executeIndexReduceScalar(launchDims, ctx.stream(), opNum, x->specialBuffer(), x->specialShapeInfo(), x->rankOf(), _extras.data(), z->specialBuffer(), z->specialShapeInfo(), z->rankOf(), _axis.data(), axis.size(), 0, reinterpret_cast<int *>(ctx.allocationBuffer()), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo(), helper.tadOffsets());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execIndexReduceOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        TadMigrationHelper helper(tad);
        dim3 launchDims = {128, 1024, 8192};

//     executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
        VectorMigrationHelper<int> _axis(axis);
        VectorMigrationHelper<T> _extras(extras);
        functions::indexreduce::IndexReduce<T>::executeIndexReduce(launchDims, ctx.stream(), opNum, x->specialBuffer(), x->specialShapeInfo(), x->rankOf(), _extras.data(), z->specialBuffer(), z->specialShapeInfo(), z->rankOf(), _axis.data(), axis.size(), 0, reinterpret_cast<int *>(ctx.allocationBuffer()), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo(), helper.tadOffsets());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execPairwiseOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<T> &extras) {
//        NativeOpExcutioner<T>::execPairwiseTransform(opNum, x->getBuffer(), x->getShapeInfo(), y->getBuffer(), y->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), extras.data());
        Nd4jPointer extraPtrs[] = {nullptr, reinterpret_cast<Nd4jPointer>(ctx.stream()), nullptr, nullptr};
        dim3 launchDims = {128, 1024, 2048};

// execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, Nd4jLong *xShapeInfo, T *y, Nd4jLong *yShapeInfo, T *result, Nd4jLong *resultShapeInfo, T *extraParams);
        VectorMigrationHelper<T> _extras(extras);

        functions::pairwise_transforms::PairWiseTransform<T>::execudaCudaShaped(launchDims, extraPtrs, opNum, x->specialBuffer(), x->specialShapeInfo(), y->specialBuffer(), y->specialShapeInfo(), z->specialBuffer(),  z->specialShapeInfo(), _extras.data());
        cudaStreamSynchronize(*ctx.stream());
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduce3ScalarOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<T> &extras) {
//        T scalar = NativeOpExcutioner<T>::execReduce3Scalar(opNum, x->buffer(), x->shapeInfo(), extras.data(), y->buffer(), y->shapeInfo());
//        z->putScalar(0, scalar);
    }

    template <typename T>
    void LegacyOpExecutor<T>::execReduce3Op(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *y, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
//        NativeOpExcutioner<T>::execReduce3(opNum, x->buffer(), x->shapeInfo(), extras.data(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()));
    }

    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}