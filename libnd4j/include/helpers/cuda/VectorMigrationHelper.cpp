//
// Created by raver on 5/25/2018.
//

#include <helpers/VectorMigrationHelper.h>
namespace nd4j {
    template <typename T>
    VectorMigrationHelper::VectorMigrationHelper(std::vector<T> &vec) {
        if (!vec.empty()) {
            auto len0 = vec.size() * sizeof(T);

            auto res0 = cudaMalloc(reinterpret_cast<void **>(&_deviceData), len0);

            if (res0 != 0)
                throw std::runtime_error("CUDA device allocation failed");

            // FIXME: we want memcpyAsync here, using shared default thread per device
            cudaMemcpy(_deviceData, vec.data(), len0, cudaMemcpyHostToDevice);
        }
    }

    VectorMigrationHelper::~VectorMigrationHelper() {
        if (_deviceData != nullptr)
            cudaFree(_deviceData);
    }

    Nd4jLong* VectorMigrationHelper::data() {
        return _deviceData;
    }

    Nd4jLong VectorMigrationHelper::size() {
        return _size;
    }
}