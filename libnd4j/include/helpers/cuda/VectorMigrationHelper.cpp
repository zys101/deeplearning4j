//
// Created by raver on 5/25/2018.
//

#include <helpers/VectorMigrationHelper.h>
#include <stdexcept>
#include <dll.h>
#include <types/float16.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace nd4j {
    template <typename T>
    VectorMigrationHelper<T>::VectorMigrationHelper(std::vector<T> &vec) {
        if (!vec.empty()) {
            _size = vec.size();
            auto len0 = vec.size() * sizeof(T);

            auto res0 = cudaMalloc(reinterpret_cast<void **>(&_deviceData), len0);

            if (res0 != 0)
                throw std::runtime_error("CUDA device allocation failed");

            // FIXME: we want memcpyAsync here, using shared default thread per device
            cudaMemcpy(_deviceData, vec.data(), len0, cudaMemcpyHostToDevice);
        }
    }

    template <typename T>
    VectorMigrationHelper<T>::~VectorMigrationHelper() {
        if (_deviceData != nullptr)
            cudaFree(_deviceData);
    }

    template <typename T>
    T* VectorMigrationHelper<T>::data() {
        return _deviceData;
    }

    template <typename T>
    Nd4jLong VectorMigrationHelper<T>::size() {
        return _size;
    }

    template class ND4J_EXPORT VectorMigrationHelper<float>;
    template class ND4J_EXPORT VectorMigrationHelper<float16>;
    template class ND4J_EXPORT VectorMigrationHelper<double>;
}