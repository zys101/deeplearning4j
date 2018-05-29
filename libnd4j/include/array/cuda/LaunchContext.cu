//
//  @author raver119@gmail.com
//

#include <array/LaunchContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace nd4j {
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
}