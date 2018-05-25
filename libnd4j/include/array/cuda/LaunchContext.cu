//
//  @author raver119@gmail.co
//

#include <array/LaunchContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace nd4j {
    void* LaunchContext::reductionPointer() {
        return reinterpret_cast<void *>(_reductionBuffer);
    }

    cudaStream_t* LaunchContext::stream() {
        return _stream;
    }

    LaunchContext* LaunchContext::setStream(cudaStream_t *stream) {
        _stream = stream;
        return this;
    }
}