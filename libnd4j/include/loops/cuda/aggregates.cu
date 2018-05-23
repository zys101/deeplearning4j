//
//  @author raver119@gmail.com
//

#include <ops/aggregate_ops.h>

#include <helpers/helper_ptrmap.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include <loops/legacy_ops.h>
#include <loops/aggregates.h>
#include <op_boilerplate.h>
#include <pointercast.h>
#include <DebugHelper.h>


template <typename T, typename OpClass>
__device__ void aggregateGeneric(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
    functions::aggregate::AggregatedFunction<T>:: template execCuda<OpClass>(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
};


template <typename T, typename OpClass>
__device__ void aggregateBatchGeneric(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {

    nd4j::PointersHelper<T> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // TODO: we probably should lift this restriction
    __shared__ int *intArrays[32];

    __shared__ T **arguments;
    __shared__ Nd4jLong **shapes;
    __shared__ int *idxArg;
    __shared__ T *realArg;

    for(int r = blockIdx.x; r < numAggregates; r += gridDim.x) {
        if (threadIdx.x == 0) {
            arguments = helper.getArguments(r);
            shapes = helper.getShapeArguments(r);
            idxArg = helper.getIndexArguments(r);
            realArg = helper.getRealArguments(r);
        }

        // we fill intArrays param in parallel within block
        if (threadIdx.x < 32 && threadIdx.x < maxIntArrays) {
            intArrays[threadIdx.x] = helper.getIntArrayArguments(r, threadIdx.x);
        }
        __syncthreads();

        functions::aggregate::AggregatedFunction<T>::template execCuda<OpClass>(arguments, helper.getNumArguments(r), shapes, helper.getNumShapeArguments(r), idxArg, helper.getNumIndexArguments(r), intArrays, helper.getNumIntArrayArguments(r), realArg, helper.getNumRealArguments(r));
    }
};



// simple aggregates
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float, INPUT(float **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, double, INPUT(double **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float16, INPUT(float16 **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float16 *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))


// batched aggregates
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float16, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, double, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

namespace functions {
    namespace aggregate {
        template <>
        _CUDA_H void AggregatedFunction<float>::execAggregateSimple(cudaStream_t *stream, dim3 &launchDims, int opNum, float **arguments,int numArguments, Nd4jLong **shapes, int numShapes,int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateSimple, float, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

            nd4j::DebugHelper::checkErrorCode(stream, "execAggregateFloat(...) failed");
        }

        template <>
        _CUDA_H void AggregatedFunction<float16>::execAggregateSimple(cudaStream_t *stream, dim3 &launchDims, int opNum, float16 **arguments,int numArguments, Nd4jLong **shapes, int numShapes,int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float16 *realArguments, int numRealArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateSimple, float16, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

            nd4j::DebugHelper::checkErrorCode(stream, "execAggregateHalf(...) failed");
        }

        template <>
        _CUDA_H void AggregatedFunction<double>::execAggregateSimple(cudaStream_t *stream, dim3 &launchDims, int opNum, double **arguments,int numArguments, Nd4jLong **shapes, int numShapes,int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateSimple, double, PARAMS(arguments, numArguments, shapes, numShapes, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

            nd4j::DebugHelper::checkErrorCode(stream, "execAggregateDouble(...) failed");
        }


        template <>
        _CUDA_H void AggregatedFunction<float>::execAggregateBatch(cudaStream_t *stream, dim3 &launchDims, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateBatchSimple, float, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void AggregatedFunction<float16>::execAggregateBatch(cudaStream_t *stream, dim3 &launchDims, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateBatchSimple, float16, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void AggregatedFunction<double>::execAggregateBatch(cudaStream_t *stream, dim3 &launchDims, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,  void *ptrToArguments) {
            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(aggregateBatchSimple, double, PARAMS(numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

            DEBUG_KERNEL(stream, opNum);
        }
    }
}