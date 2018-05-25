//
// Created by raver on 5/25/2018.
//

#include <helpers/TadMigrationHelper.h>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.api>

namespace nd4j {
    TadMigrationHelper::TadMigrationHelper(shape::TAD &tad) {
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        auto len0 = shape::shapeInfoByteLength(tad.tadOnlyShapeInfo);
        auto len1 = tad.numTads * sizeof(Nd4jLong);

        auto res0 = cudaMalloc(reinterpret_cast<void **>(&_deviceTadShapeInfo), len0);
        auto res1 = cudaMalloc(reinterpret_cast<void **>(&_deviceTadOffsets), len1);

        if (res0 != 0 || res1 != 0)
            throw std::runtime_error("CUDA device allocation failed");

        // FIXME: we want memcpyAsync here, using shared default thread per device
        cudaMemcpy(_deviceTadShapeInfo, tad.tadOnlyShapeInfo, len0, cudaMemcpyHostToDevice);
        cudaMemcpy(_deviceTadOffsets, tad.tadOffsets, len1, cudaMemcpyHostToDevice);
    }

    TadMigrationHelper::~TadMigrationHelper() {
        if (_deviceTadOffsets != nullptr)
            cudaFree(_deviceTadOffsets);


        if (_deviceTadShapeInfo != nullptr)
            cudaFree(_deviceTadShapeInfo);
    }

    Nd4jLong* TadMigrationHelper::tadShapeInfo() {
        return _deviceTadShapeInfo;
    }

    Nd4jLong* TadMigrationHelper::tadOffsets() {
        return _deviceTadOffsets;
    }
}