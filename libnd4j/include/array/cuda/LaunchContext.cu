//
//  @author raver119@gmail.com
//

#include <array/LaunchContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <helpers/logger.h>

namespace nd4j {
    LaunchContext::LaunchContext() {
        // default constructor, just to make clang/ranlib happy
        //ALLOCATE(_stream, _workspace, sizeof(cudaStream_t), cudaStream_t);
        _stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
        cudaStreamCreate(&_stream[0]);
        // 10M bytes of device memory for reductionBuffer
        ALLOCATE_SPECIAL(_reductionBuffer, _workspace, 1250000 * sizeof(Nd4jLong), Nd4jLong) ;
        // 16 bytes of device memory for scallarPointer
        ALLOCATE_SPECIAL(_scalarPointer, _workspace, 2 * sizeof(Nd4jLong), Nd4jLong) ;
        // 10M bytes of device memory for allocationBuffer
        ALLOCATE_SPECIAL(_allocationBuffer, _workspace, 1250000 * sizeof(Nd4jLong), Nd4jLong) ;
    }

    LaunchContext::~LaunchContext() {
        // default constructor, just to make clang/ranlib happy
        if (_stream)
            cudaStreamDestroy(*_stream);
        //RELEASE(_stream, _workspace);
        RELEASE_SPECIAL(_allocationBuffer, _workspace);
        RELEASE_SPECIAL(_reductionBuffer, _workspace);
        RELEASE_SPECIAL(_scalarPointer, _workspace);
        free(_stream);
    }

    void* LaunchContext::reductionPointer() {
        return reinterpret_cast<void *>(_reductionBuffer);
    }

    void* LaunchContext::allocationBuffer() {
        return reinterpret_cast<void *>(_allocationBuffer);
    }

    cudaStream_t* LaunchContext::stream() {
        return _stream;
    }

    LaunchContext* LaunchContext::setCudaStream(cudaStream_t *stream) {
        _stream = stream;
        return this;
    }
    LaunchContext* LaunchContext::defaultContext() {
        /**
         * defaultContext should be platform-specific
         */
        return nullptr;
    }

}