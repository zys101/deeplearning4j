//
//  @author raver119@gmail.com
//

#include <array/LaunchContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <helpers/logger.h>

//static cudaStream_t defaultStream;
namespace nd4j {
    LaunchContext::LaunchContext() {
        // default constructor, just to make clang/ranlib happy
        _stream = new cudaStream_t;
        cudaStreamCreate(_stream);
    }

    LaunchContext::~LaunchContext() {
        // default constructor, just to make clang/ranlib happy
        cudaStreamDestroy(*_stream);
        delete _stream;
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