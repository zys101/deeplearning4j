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
    void LegacyOpExecutor<T>::execReduceOp(nd4j::LaunchContext &ctx, int opNum, NDArray<T> *x, NDArray<T> *z, std::vector<int> &axis, std::vector<T> &extras) {
        shape::TAD tad(x->getShapeInfo(), axis.data(), static_cast<int>(axis.size()));
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        TadMigrationHelper helper(tad);

        //NativeOpExcutioner<T>::execReduce(opNum, x->buffer(), x->shapeInfo(), extras.data(), z->buffer(), z->shapeInfo(), axis.data(), static_cast<int>(axis.size()), tad.tadOnlyShapeInfo, tad.tadOffsets);

        dim3 launchDims = {128, 1024, 8192};

        VectorMigrationHelper<int> _axis(axis);
        VectorMigrationHelper<T> _extras(extras);

        functions::reduce::ReduceFunction<T>::execReduceXD(launchDims, ctx.stream(), opNum, x->rankOf(), x->specialBuffer(), x->specialShapeInfo(), _extras.data(), z->specialBuffer(), z->specialShapeInfo(), _axis.data(), axis.size(), reinterpret_cast<T *>(ctx.reductionPointer()), helper.tadShapeInfo(), helper.tadOffsets());

        cudaStreamSynchronize(*ctx.stream());
    }


    template class ND4J_EXPORT LegacyOpExecutor<float>;
    template class ND4J_EXPORT LegacyOpExecutor<float16>;
    template class ND4J_EXPORT LegacyOpExecutor<double>;
}