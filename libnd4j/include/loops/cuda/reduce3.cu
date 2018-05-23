//
//
//

#include <op_boilerplate.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <pointercast.h>
#include <helpers/helper_ptrmap.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include <loops/legacy_ops.h>
#include <loops/reduce3.h>

#include <DebugHelper.h>


namespace functions {
    namespace reduce3 {

        template <typename T>
        void _CUDA_D Reduce3<T>::exec(
                const int opNum,
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationPointer,
                UnifiedSharedMemory *manager,
                Nd4jLong *tadOnlyShapeInfo,
                Nd4jLong *tadOffsets,
                Nd4jLong *yTadOnlyShapeInfo,
                Nd4jLong *yTadOffsets) {
            DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
        };

        template<typename T>
        template<typename OpType>
        void _CUDA_D Reduce3<T>::aggregatePartials(T **sPartialsRef, Nd4jLong tid, Nd4jLong numItems, T *extraParamsRef) {
            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.
            T *sPartials = *sPartialsRef;
            Nd4jLong floorPow2 = numItems;

            if (floorPow2 & (floorPow2 - 1)) {
                while (floorPow2 & (floorPow2 - 1)) {
                    floorPow2 &= floorPow2 - 1;
                }
                if (tid >= floorPow2) {
                    sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParamsRef);
                }
                __syncthreads();
            }

            for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads) {
                    sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParamsRef);
                }
                __syncthreads();
            }
        };


        template <typename T>
        template <typename OpType>
        void _CUDA_D Reduce3<T>::execScalarCuda(
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo, int *allocationPointer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {

            T *sPartials = (T *) manager->getSharedReductionBuffer(); // val.getPointer();

            // FIXME: this ugly fast fix.
            __shared__ T extraZ[3];
            if (threadIdx.x == 0) {
                extraZ[0] = (T) 0.0f;
                extraZ[1] = (T) 0.0f;

                if (extraParams != NULL) {
                    extraZ[2] = extraParams[0];
                } else extraZ[2] = (T) 0.0f;
            }

            __syncthreads();

            T startingVal = OpType::startingValue(dx);
            Nd4jLong length = shape::length(xShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
            int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            char xOrder = shape::order(xShapeInfo);
            char yOrder = shape::order(yShapeInfo);

            if(xOrder == yOrder && (xElementWiseStride > 0 && yElementWiseStride > 0) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {
                if (xElementWiseStride == 1 && yElementWiseStride == 1) {
                    for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
                        startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i], dy[i], extraZ), extraZ);
                    }
                }
                else {
                    for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
                        startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i * xElementWiseStride], dy[i * yElementWiseStride], extraZ), extraZ);
                    }
                }

                sPartials[threadIdx.x] = startingVal;
            } else {
                __shared__ Nd4jLong *xShape;
                __shared__ Nd4jLong *yShape;
                __shared__ Nd4jLong *xStride;
                __shared__ Nd4jLong *yStride;
                __shared__ int rank;
                if (threadIdx.x == 0) {

                    xShape = shape::shapeOf(xShapeInfo);
                    yShape = shape::shapeOf(yShapeInfo);
                    xStride = shape::stride(xShapeInfo);
                    yStride = shape::stride(yShapeInfo);
                    rank = shape::rank(xShapeInfo);
                }
                __syncthreads();
                T startingVal = OpType::startingValue(dx);

                T *sPartials = (T *) manager->getSharedReductionBuffer();

                Nd4jLong xCoords[MAX_RANK];
                Nd4jLong yCoords[MAX_RANK];

                sPartials[threadIdx.x] = startingVal;

                for(Nd4jLong i = tid ;i < length; i += gridDim.x * blockDim.x) {
                    shape::ind2subC(rank,xShape,i,xCoords);
                    shape::ind2subC(rank,yShape,i,yCoords);

                    auto offset = shape::getOffset(0, xShape, xStride, xCoords,rank);
                    auto yOffset = shape::getOffset(0,yShape, yStride, yCoords,rank);

                    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[offset], dy[yOffset], extraZ), extraZ);
                }
            }

            __syncthreads();

            T **sPartialsRef = (T **) &sPartials;
            aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, length), extraZ);

            __syncthreads();
            if (gridDim.x > 1) {
                unsigned int *tc = (unsigned int *)reductionBuffer;
                __shared__ bool amLast;
                int rank = shape::rank(xShapeInfo);
                tid = threadIdx.x;
                T *extraBuffer = (T *) allocationPointer;
                if (threadIdx.x == 0) {
                    reductionBuffer[blockIdx.x] = sPartials[0];
                    extraBuffer[blockIdx.x] = extraZ[0];
                    extraBuffer[gridDim.x + blockIdx.x] = extraZ[1];
                }
                __threadfence();
                __syncthreads();

                if (threadIdx.x == 0) {
                    unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                    amLast = (ticket == gridDim.x - 1);
                }

                sPartials[tid] = startingVal;
                __syncthreads();

                if (amLast) {
                    tc[16384] = 0;

                    sPartials[threadIdx.x] = OpType::startingValue(dx);

                    // TODO: later probably replace this. Right now we need extraZ sync for CosineSimilarity ONLY
                    if (tid == 0 && extraZ[0] != (T) 0.0 && extraZ[1] != (T) 0.0) {
                        extraZ[0] = 0.0;
                        extraZ[1] = 0.0;
                        for (int i = 0; i < gridDim.x; i++) {
                            extraZ[0] += extraBuffer[i];
                            extraZ[1] += extraBuffer[gridDim.x + i];
                        }
                    }

                    for (Nd4jLong i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
                        sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraZ);
                    }
                    __syncthreads();

                    aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraZ);

                    __syncthreads();
                    if (threadIdx.x == 0) {
                        result[0] = OpType::postProcess(sPartials[0], length, extraZ);
                    }
                }
            } else {
                if (tid == 0) {
                    unsigned int *tc = (unsigned *)reductionBuffer;
                    tc[16384] = 0;

                    result[0] = OpType::postProcess(sPartials[0], length, extraZ);
                }
            }
        };

        template <typename T>
        template <typename OpType>
        void _CUDA_D Reduce3<T>::transformAll(
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationPointer,
                UnifiedSharedMemory *manager,
                Nd4jLong *xTadShapeInfo,
                Nd4jLong *xOffsets,
                Nd4jLong *yTadShapeInfo,
                Nd4jLong *yOffsets) {

            // initialize partials first
            T *sPartials = (T *) manager->getSharedReductionBuffer();
            T startingVal = OpType::startingValue(dx);
            sPartials[threadIdx.x] = startingVal;
            T *tempX = sPartials + blockDim.x;

            const int maxBlock = blockDim.x;

            __shared__ T extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

            __shared__ int xTadLength;
            __shared__ int yTadLength;

            __shared__ int xTads;
            __shared__ int yTads;

            __shared__ Nd4jLong *xShape;
            __shared__ Nd4jLong *xStride;
            __shared__ int xRank;

            __shared__ Nd4jLong *yShape;
            __shared__ Nd4jLong *yStride;
            __shared__ int yRank;

            //reading initial data
            if (threadIdx.x == 0) {
                xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

                xTads = shape::length(xShapeInfo) / xTadLength;
                yTads = shape::length(yShapeInfo) / yTadLength;

                xShape = shape::shapeOf(xTadShapeInfo);
                xStride = shape::stride(xTadShapeInfo);
                xRank = shape::rank(xTadShapeInfo);

                yShape = shape::shapeOf(yTadShapeInfo);
                yStride = shape::stride(yTadShapeInfo);
                yRank = shape::rank(yTadShapeInfo);
            }
            __syncthreads();


            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong yCoord[MAX_RANK];


            int limit = xTadLength / maxBlock;
            if (xTadLength % maxBlock > 0)
                limit++;


            for (int r = blockIdx.x; r < xTads; r += blockDim.x * gridDim.x) {
                T *x = dx + xOffsets[r];

                if (threadIdx.x < xTadLength && threadIdx.x < maxBlock) {
                    if (shape::order(xTadShapeInfo) == 'c') {
                        shape::ind2subC(xRank, xShape, threadIdx.x, xCoord);
                    } else {
                        shape::ind2sub(xRank, xShape, threadIdx.x, xCoord);
                    }

                    auto xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                    tempX[threadIdx.x] = x[xO];
                }

                for (int g = 0; g < yTads; g++) {
                    T *y = dy + yOffsets[g];

                    int ri = (r * yTads) + g;

                    sPartials[threadIdx.x] = startingVal;
                    if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
                        extraZ[threadIdx.x] = (T) startingVal;
                    }
                    __syncthreads();

                    // we might have data too large for single cache block, rendering cache useless though :(
                    for (int t = 0; t < limit; t++) {

                        // we reset tempX IF we have >1 tiles
                        if (t >= 1 || (limit > 1 && g > 0))
                            if (threadIdx.x + (t * maxBlock) < xTadLength) {
                                if (shape::order(xTadShapeInfo) == 'c') {
                                    shape::ind2subC(xRank, xShape, threadIdx.x + (t * maxBlock), xCoord);
                                } else {
                                    shape::ind2sub(xRank, xShape, threadIdx.x + (t * maxBlock), xCoord);
                                }

                                Nd4jLong xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                                tempX[threadIdx.x] = x[xO];
                                //                                tempX[threadIdx.x] = x[threadIdx.x + (t * maxBlock)];
                            }

                        for (int f = threadIdx.x + (t * maxBlock); f < xTadLength && f < threadIdx.x + ((t + 1) * maxBlock); f += blockDim.x * gridDim.x) {
                            if (shape::order(yTadShapeInfo) == 'c') {
                                shape::ind2subC(yRank, yShape, f, yCoord);
                            } else {
                                shape::ind2sub(yRank, yShape, f, yCoord);
                            }

                            Nd4jLong yO = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(tempX[threadIdx.x], y[yO], extraZ), extraZ);
                        }

                        // we MUST step through this block altogether
                        __syncthreads();
                    }

                    T **sPartialsRef = (T **) &sPartials;
                    aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, xTadLength), extraZ);

                    __syncthreads();

                    if (threadIdx.x == 0) {
                        result[ri] = OpType::postProcess(sPartials[threadIdx.x],xTadLength, extraZ);
                    }

                    __syncthreads();
                }
            }
        };


        template <typename T>
        template <typename OpType>
        void _CUDA_D Reduce3<T>::transform(
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationPointer,
                UnifiedSharedMemory *manager,
                Nd4jLong *tadOnlyShapeInfo,
                Nd4jLong *tadOffsets,
                Nd4jLong *yTadOnlyShapeInfo,
                Nd4jLong *yTadOffsets) {
            /**
             * Gpu information for the problem
             */
            int tid = threadIdx.x + blockIdx.x * blockDim.x;

            __shared__ int resultScalar;

            __shared__ int xElementWiseStride;
            __shared__ int yElementWiseStride;
            //shared memory space for storing intermediate results
            //SharedMemory <T> val;
            T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
            T init = OpType::startingValue(dx);
            sPartials[threadIdx.x] = init;

            __shared__ T extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

            //length for the tad

            __shared__ Nd4jLong resultLength;
            __shared__ int tadLength;
            __shared__ int yLength;
            __shared__ int tadElementWiseStride;
            __shared__ int yTadElementWiseStride;

            T startingVal = OpType::startingValue(dx);

            T reduction = OpType::startingValue(dx);
            if (threadIdx.x == 0) {
                if (resultShapeInfo != nullptr)
                    resultLength = shape::length(resultShapeInfo);
                else resultLength = 1;

                if (dimensionLength == 1) {
                    if (dimension == nullptr || dimension[0] == MAX_DIMENSION)
                        resultScalar = 1;
                    else
                        resultScalar = 0;
                }
                else
                    resultScalar = 0;

                if (resultLength == 1)
                    resultScalar = 1;

                auto xStride = shape::stride(xShapeInfo);
                char xOrder = shape::order(xShapeInfo);

                tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                tadElementWiseStride = shape::elementWiseStride(tadOnlyShapeInfo);
                yLength = shape::length(yShapeInfo);

                if (yTadOnlyShapeInfo != nullptr)
                    yTadElementWiseStride = shape::elementWiseStride(yTadOnlyShapeInfo);
            }
            __syncthreads();

            // code branch for TAD vs full array
            if (tadLength == yLength) {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                auto yShape = shape::shapeOf(yShapeInfo);
                auto yStride = shape::stride(yShapeInfo);
                auto xShape = shape::shapeOf(tadOnlyShapeInfo);
                auto xStride = shape::stride(tadOnlyShapeInfo);
                int yRank = shape::rank(yShapeInfo);
                int xRank = shape::rank(tadOnlyShapeInfo);


                for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
                    int xOffsetForTad = tadOffsets[i];

                    if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
                        extraZ[threadIdx.x] = (T) startingVal;
                    }
                    __syncthreads();

                    for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                        shape::ind2subC(xRank,xShape, j, xCoord);
                        shape::ind2subC(yRank,yShape, j, yCoord);

                        Nd4jLong xOffset = shape::getOffset(xOffsetForTad, xShape, xStride, xCoord, xRank);
                        Nd4jLong yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                        sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ), extraZ);
                    }
                    __syncthreads();

                    T **sPartialsRef = (T **) &sPartials;
                    aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                    __syncthreads();
                    if (threadIdx.x == 0)
                        result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

                    __syncthreads();
                }
            } else  if (!resultScalar) {
                if(tadElementWiseStride >= 1 && yTadElementWiseStride) {
                    for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
                        int xOffsetForTad = tadOffsets[i];
                        int yOffsetForTad = yTadOffsets[i];

                        if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
                            extraZ[threadIdx.x] = (T) startingVal;
                        }
                        __syncthreads();

                        if (threadIdx.x < tadLength)
                            sPartials[threadIdx.x] =  OpType::op(dx[xOffsetForTad + tadElementWiseStride * threadIdx.x],dy[yOffsetForTad + yTadElementWiseStride * threadIdx.x], extraZ);

                        for(int j = threadIdx.x + blockDim.x; j < tadLength; j += blockDim.x) {
                            sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffsetForTad + tadElementWiseStride * j],dy[yOffsetForTad + yTadElementWiseStride * j], extraZ), extraZ);
                        }
                        __syncthreads();

                        T **sPartialsRef = (T **) &sPartials;
                        aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                        __syncthreads();
                        if (threadIdx.x == 0)
                            result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

                        __syncthreads();
                    }
                }
                else {
/*
						// DO NOT REMOVE THIS COMMENTED BLOCK PLEASE

						for (int r = blockIdx.x; r < tad->numTads; r += gridDim.x) {
                            if (threadIdx.x == 0)
                                tad->createOffsetForBlock(r);
                            __syncthreads();

                            int tadOffsetForBlock = tad->tadOffsetForBlock;
                            T *xVal = dx + tadOffsetForBlock;


                            sPartials[threadIdx.x] = this->startingValue(xVal);
                            for(int i = threadIdx.x; i < tad->tadLength; i+= blockDim.x) {
                    			int xOffsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength, nullptr);
								int yOffsetForTad = shape::tadOffset(i, yShapeInfo, dimension, dimensionLength, nullptr);

                                sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],dx[tadOffsetForBlock + i *  tad->tadElementWiseStride], extraParams);
                            }
                            __syncthreads();

                            // aggregate. do NOT reduce for elements > tadLength
                            T **sPartialsRef = (T **) &sPartials;
                            aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength), extraParams);


                            __syncthreads();
                            if (threadIdx.x == 0)
                                result[r] = this->postProcess(sPartials[threadIdx.x], tad->tadLength, extraParams);
                        }

*/

                    Nd4jLong xCoord[MAX_RANK];
                    Nd4jLong yCoord[MAX_RANK];

                    auto yShape = shape::shapeOf(yTadOnlyShapeInfo);
                    auto yStride = shape::stride(yTadOnlyShapeInfo);
                    auto xShape = shape::shapeOf(tadOnlyShapeInfo);
                    auto xStride = shape::stride(tadOnlyShapeInfo);
                    int yRank = shape::rank(yTadOnlyShapeInfo);
                    int xRank = shape::rank(tadOnlyShapeInfo);


                    for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
                        auto xOffsetForTad = tadOffsets[i];
                        auto yOffsetForTad = yTadOffsets[i];

                        if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
                            extraZ[threadIdx.x] = (T) startingVal;
                        }
                        __syncthreads();

                        for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                            shape::ind2subC(xRank,xShape, j, xCoord);
                            shape::ind2subC(yRank,yShape, j, yCoord);

                            auto xOffset = shape::getOffset(xOffsetForTad, xShape, xStride, xCoord, xRank);
                            auto yOffset = shape::getOffset(yOffsetForTad, yShape, yStride, yCoord, yRank);

                            sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ), extraZ);
                        }
                        __syncthreads();

                        T **sPartialsRef = (T **) &sPartials;
                        aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                        __syncthreads();
                        if (threadIdx.x == 0)
                            result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

                        __syncthreads();
                    }

                }
            }
        };


        template <typename T>
        void Reduce3<T>::execAllCuda(
                const int opNum,
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationPointer,
                UnifiedSharedMemory *manager,
                Nd4jLong *tadOnlyShapeInfo,
                Nd4jLong *tadOffsets,
                Nd4jLong *yTadOnlyShapeInfo,
                Nd4jLong *yTadOffsets) {
            DISPATCH_BY_OPNUM(transformAll, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
        };

        template <typename T>
        void _CUDA_D Reduce3<T>::execScalarCuda(
                const int opNum,
                T *dx,
                Nd4jLong *xShapeInfo,
                T *dy,
                Nd4jLong *yShapeInfo,
                T *extraParams,
                T *result,
                Nd4jLong *resultShapeInfo,
                int * allocationPointer,
                T *reductionBuffer,
                UnifiedSharedMemory *manager,
                Nd4jLong *tadOnlyShapeInfo) {
            DISPATCH_BY_OPNUM(execScalarCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, allocationPointer, reductionBuffer, manager, tadOnlyShapeInfo), REDUCE3_OPS);
        };
    }
}



/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post
 */
template <typename T>
__device__ void reduce3Generic(
        const int opNum,
        T *dx,
        Nd4jLong *xShapeInfo,
        T *dy,
        Nd4jLong *yShapeInfo,
        T *extraParams,
        T *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

    }
    __syncthreads();

    functions::reduce3::Reduce3<T>::exec(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot,
            allocationPointer,
            manager,
            tadOnlyShapeInfo,
            tadOffsets,
            yTadOnlyShapeInfo,
            yTadOffsets);
}

template <typename T>
__device__ void reduce3AllGeneric(
        const int opNum,
        T *dx,
        Nd4jLong *xShapeInfo,
        T *dy,
        Nd4jLong *yShapeInfo,
        T *extraParams,
        T *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot,
        int *allocationPointer,
        Nd4jLong *tadOnlyShapeInfo,
        Nd4jLong *tadOffsets,
        Nd4jLong *yTadOnlyShapeInfo,
        Nd4jLong *yTadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

    }
    __syncthreads();

    functions::reduce3::Reduce3<T>::execAllCuda(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot,
            allocationPointer,
            manager,
            tadOnlyShapeInfo,
            tadOffsets,
            yTadOnlyShapeInfo,
            yTadOffsets);
}

template <typename T>
__device__ void reduce3ScalarGeneric(
        int opNum,
        T *dx,
        Nd4jLong *xShapeInfo,
        T *dy,
        Nd4jLong *yShapeInfo,
        T *extraParams,
        T *result,
        Nd4jLong *resultShapeInfo, int *allocationPointer,
        T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }
    __syncthreads();

    functions::reduce3::Reduce3<T>::execScalarCuda(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            allocationPointer,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo);
}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */

__global__ void reduce3Double(
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *dy,
        Nd4jLong *yShapeInfo,
        double *extraParams,
        double *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3Generic<double>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3AllDouble(
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *dy,
        Nd4jLong *yShapeInfo,
        double *extraParams,
        double *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3AllGeneric<double>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */

__global__ void reduce3Float(
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *dy,
        Nd4jLong *yShapeInfo,
        float *extraParams,
        float *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3Generic<float>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}


__global__ void reduce3AllFloat(
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *dy,
        Nd4jLong *yShapeInfo,
        float *extraParams,
        float *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3AllGeneric<float>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3Half(
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *dy,
        Nd4jLong *yShapeInfo,
        float16 *extraParams,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3Generic<float16>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3AllHalf(
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *dy,
        Nd4jLong *yShapeInfo,
        float16 *extraParams,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3AllGeneric<float16>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3ScalarFloat(
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *dy,
        Nd4jLong *yShapeInfo,
        float *extraParams,
        float *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3ScalarGeneric<float>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo, allocationPointer,
            reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3ScalarHalf(
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *dy,
        Nd4jLong *yShapeInfo,
        float16 *extraParams,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3ScalarGeneric<float16>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo, allocationPointer,
            reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

__global__ void reduce3ScalarDouble(
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *dy,
        Nd4jLong *yShapeInfo,
        double *extraParams,
        double *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        int postProcessOrNot, int *allocationPointer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
    reduce3ScalarGeneric<double>(
            opNum,
            dx,
            xShapeInfo,
            dy,
            yShapeInfo,
            extraParams,
            result,
            resultShapeInfo, allocationPointer,
            reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}