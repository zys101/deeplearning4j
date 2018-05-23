/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#define EXTRA_PARAMS_LENGTH 10

#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/sharedmem.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>
#include <helpers/shape.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"

#ifdef __CUDACC__
void __global__ reduce3Double(
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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

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
        int postProcessOrNot, int *allocationPointer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


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
        int postProcessOrNot, int *allocationPointer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


#endif

namespace functions {
    namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
        template<typename T>
        class Reduce3 {

        public:
#ifdef __CUDACC__
            virtual __device__ inline T opAtomic(T d1, T d2, T *extraParamsRef) = 0;

     /**
     * Aggregate shared memory
     * @param sPartialsRef
     * @param tid
     * @param extraParams
     */

     template<typename OpType>
     static _CUDA_D void aggregatePartials(T **sPartialsRef, Nd4jLong tid, Nd4jLong numItems, T *extraParamsRef);


     template<typename OpType>
     static _CUDA_D void execScalarCuda(T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *allocationPointer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

     template<typename OpType>
     static _CUDA_D void transformAll(
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
					Nd4jLong *yOffsets);

     template<typename OpType>
     static _CUDA_D void transform(
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
					Nd4jLong *yTadOffsets);

            static _CUDA_D void exec(
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
				Nd4jLong *yTadOffsets);



            static _CUDA_D void execAllCuda(
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
				Nd4jLong *yTadOffsets);


			static _CUDA_D void execScalarCuda(
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
				Nd4jLong *tadOnlyShapeInfo);
#endif



            static _CUDA_H T execScalar(
                    const int opNum,
                    T *x,
                    Nd4jLong *xShapeInfo,
                    T *extraParamsVals,
                    T *y,
                    Nd4jLong *yShapeInfo) {
                RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x,
                                                               xShapeInfo,
                                                               extraParamsVals,
                                                               y,
                                                               yShapeInfo), REDUCE3_OPS);
            }

            static void exec( const int opNum,
                              T *x, Nd4jLong *xShapeInfo,
                              T *extraParamsVals,
                              T *y,
                              Nd4jLong *yShapeInfo,
                              T *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParamsVals,
                                               y, yShapeInfo,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength), REDUCE3_OPS);
            }


            static void exec( const int opNum,
                              T *x, Nd4jLong *xShapeInfo,
                              T *extraParamsVals,
                              T *y,
                              Nd4jLong *yShapeInfo,
                              T *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParamsVals,
                                               y, yShapeInfo,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength, tadShapeInfo, tadOffsets), REDUCE3_OPS);
            }

            static void execAll( const int opNum,
                                 T *x,
                                 Nd4jLong *xShapeInfo,
                                 T *extraParamsVals,
                                 T *y,
                                 Nd4jLong *yShapeInfo,
                                 T *result,
                                 Nd4jLong *resultShapeInfoBuffer,
                                 int *dimension,
                                 int dimensionLength,
                                 Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
                                 Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {
                DISPATCH_BY_OPNUM(execAll, PARAMS(x,
                                                  xShapeInfo,
                                                  extraParamsVals,
                                                  y, yShapeInfo,
                                                  result,
                                                  resultShapeInfoBuffer,
                                                  dimension,
                                                  dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), REDUCE3_OPS);
            }



            template<typename OpType>

            static _CUDA_H T execScalar(
                    T *x,
                    Nd4jLong *xShapeInfo,
                    T *extraParams,
                    T *y,
                    Nd4jLong *yShapeInfo) {
                T startingVal = OpType::startingValue(x);
                auto length = shape::length(xShapeInfo);
                auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                auto yElementWiseStride = shape::elementWiseStride(yShapeInfo);

                T extraParamsVals[3] = {(T) 0.0, (T) 0.0, (T) 0.0};
                // it's possible case for EqualsWithEps op
                if (extraParams != nullptr) {
                    extraParamsVals[2] = extraParams[0];
                }


                char xOrder = shape::order(xShapeInfo);
                char yOrder = shape::order(yShapeInfo);
                if(xOrder == yOrder && (xElementWiseStride  >=1 && yElementWiseStride >= 1) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {
                    if (xElementWiseStride == 1 && yElementWiseStride == 1) {

// TODO:: proper reduction required here
                        for(int i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal,
                                                         OpType::op(x[i],y[i],
                                                                    extraParamsVals),
                                                         extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);

                    }

                    else {
// TODO:: proper reduction required here
                        for(Nd4jLong i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal, OpType::op(x[i * xElementWiseStride],y[i * yElementWiseStride], extraParamsVals), extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);
                    }

                }


                else {
                    Nd4jLong xCoords[MAX_RANK];
                    Nd4jLong yCoords[MAX_RANK];

                    int xRank = shape::rank(xShapeInfo);
                    int yRank = shape::rank(yShapeInfo);

                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    auto yShape = shape::shapeOf(yShapeInfo);
                    auto yStride = shape::stride(yShapeInfo);

                    for(unsigned int i = 0 ;i < length; i++) {
                        shape::ind2subC(xRank, xShape, i, xCoords);
                        shape::ind2subC(yRank, yShape, i, yCoords);

                        auto offset = shape::getOffset(0, xShape, xStride, xCoords, xRank);
                        auto yOffset = shape::getOffset(0, yShape, yStride, yCoords, yRank);

                        startingVal = OpType::update(startingVal, OpType::op(x[offset], y[yOffset], extraParamsVals), extraParamsVals);
                    }
                }

                return OpType::postProcess(startingVal, length, extraParamsVals);;
            }


            template<typename OpType>
            static void execAll(
                    T *x,
                    Nd4jLong *xShapeInfo,
                    T *extraParams,
                    T *y,
                    Nd4jLong *yShapeInfo,
                    T *result,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

                auto xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

                auto xTads = shape::length(xShapeInfo) / xTadLength;
                auto yTads = shape::length(yShapeInfo) / yTadLength;

                auto xShape = shape::shapeOf(xTadShapeInfo);
                auto xStride = shape::stride(xTadShapeInfo);
                int xRank = shape::rank(xTadShapeInfo);

                auto yShape = shape::shapeOf(yTadShapeInfo);
                auto yStride = shape::stride(yTadShapeInfo);
                int yRank = shape::rank(yTadShapeInfo);


                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                T startingVal = OpType::startingValue(x);

#pragma  omp parallel for proc_bind(AFFINITY) default(shared) private(xCoord, yCoord)
                for (Nd4jLong r = 0; r < xTads; r++) {
                    Nd4jLong xOffset = xOffsets[r];

                    T *lX = x + xOffset;

                    for (Nd4jLong g = 0; g < yTads; g++) {
                        auto yOffset = yOffsets[g];
                        T *lY = y + yOffset;

                        auto ri = (r * yTads) + g;

                        T *localExtraParams = nullptr;
                        if (OpType::extraParamsLen > 0)
                            localExtraParams = new T[OpType::extraParamsLen];
                        for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                            localExtraParams[extraParamsIdx] = startingVal;
                        }

                        for (int f = 0; f < xTadLength; f++) {
                            if (shape::order(yTadShapeInfo) == 'c') {
                                shape::ind2subC(yRank, yShape, f, yCoord);
                            } else {
                                shape::ind2sub(yRank, yShape, f, yCoord);
                            }

                            if (shape::order(xTadShapeInfo) == 'c') {
                                shape::ind2subC(xRank, xShape, f, xCoord);
                            } else {
                                shape::ind2sub(xRank, xShape, f, xCoord);
                            }

                            auto xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto yO = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                            result[ri] = OpType::update(result[ri], OpType::op(lX[xO], lY[yO], localExtraParams), localExtraParams);
                        }

                        result[ri] = OpType::postProcess(result[ri], xTadLength, localExtraParams);

                        if (localExtraParams != nullptr)
                            delete[] localExtraParams;
                    }
                }

            }


            template<typename OpType>
            static void exec(
                    T *x,
                    Nd4jLong *xShapeInfo,
                    T *extraParams,
                    T *y,
                    Nd4jLong *yShapeInfo,
                    T *result,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
                T startingVal = OpType::startingValue(x);

                auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto tads = shape::length(xShapeInfo) / tadLength;

                auto *xShape = shape::shapeOf(tadShapeInfo);
                auto *xStride = shape::stride(tadShapeInfo);
                int xRank = shape::rank(tadShapeInfo);

                auto *yShape = shape::shapeOf(yShapeInfo);
                auto *yStride = shape::stride(yShapeInfo);
                int yRank = shape::rank(yShapeInfo);

                //shape::printShapeInfoLinear(xShapeInfo);
                //shape::printShapeInfoLinear(yShapeInfo);
                //shape::printShapeInfoLinear(resultShapeInfoBuffer);
                //shape::printShapeInfoLinear(tadShapeInfo);

                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

//#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
                for (Nd4jLong r = 0; r < tads; r++) {
                    Nd4jLong offset = tadOffsets[r];

                    T *localExtraParams = nullptr;
                    if (OpType::extraParamsLen > 0)
                        localExtraParams = new T[OpType::extraParamsLen];
                    for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                        localExtraParams[extraParamsIdx] = startingVal;
                    }

                    for (Nd4jLong f = 0; f < tadLength; f++) {
                        if (shape::order(tadShapeInfo) == 'c') {
                            shape::ind2subC(xRank, xShape, f, xCoord);
                            shape::ind2subC(yRank, yShape, f, yCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, f, xCoord);
                            shape::ind2sub(yRank, yShape, f, yCoord);
                        }

                        Nd4jLong xOffset = shape::getOffset(offset, xShape, xStride, xCoord, xRank);
                        Nd4jLong yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                        result[r] = OpType::update(result[r], OpType::op(x[xOffset], y[yOffset], localExtraParams), localExtraParams);
                    }

                    result[r] = OpType::postProcess(result[r], tadLength, localExtraParams);

                    if (localExtraParams != nullptr)
                        delete[] localExtraParams;
                }
            }

            template<typename OpType>
            static void exec(
                    T *x,
                    Nd4jLong *xShapeInfo,
                    T *extraParams,
                    T *y,
                    Nd4jLong *yShapeInfo,
                    T *result,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength) {
                T extraParamsVals[3] = {(T) 0.0, (T) 0.0, (T) 0.0};


                if(shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar<OpType>(
                            x,
                            xShapeInfo,
                            extraParamsVals,
                            y,
                            yShapeInfo);
                    return;
                }



                char xOrder = shape::order(xShapeInfo);
                char yOrder = shape::order(yShapeInfo);
                if(xOrder != yOrder) {
                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong yStridesIter[MAX_RANK];

                    auto xShape = shape::shapeOf(xShapeInfo);

                    auto xStride = shape::stride(xShapeInfo);
                    auto yStride = shape::stride(yShapeInfo);

                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 x,
                                                 xStride,
                                                 y,
                                                 yStride,
                                                 &rank,
                                                 shapeIter,
                                                 &x,
                                                 xStridesIter,
                                                 &y,
                                                 yStridesIter) >= 0) {

                        Nd4jLong resultLength = shape::length(resultShapeInfoBuffer);
                        Nd4jLong tadLength = shape::tadLength(xShapeInfo,dimension,dimensionLength);
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                Nd4jLong xOffset = shape::getOffset(0,xShape,xStride,coord,rank);
                                auto reductionIndex = xOffset / resultLength;
                                result[reductionIndex] = OpType::update(result[reductionIndex], OpType::op(x[0],y[0], extraParamsVals), extraParamsVals);
                            } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     x,
                                                     xStridesIter,
                                                     y,
                                                     yStridesIter);


//#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
                        for(Nd4jLong i = 0; i < resultLength ;i++) {
                            result[i] = OpType::postProcess(result[i],tadLength, extraParamsVals);
                        }
                    }

                    else {
                        printf("Unable to prepare array\n");
                    }
                }
                else {
                    T startingVal = OpType::startingValue(x);

                    Nd4jLong resultLength = shape::length(resultShapeInfoBuffer);
                    shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                    xTad.createTadOnlyShapeInfo();
                    xTad.createOffsets();


                    shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                    yTad.createTadOnlyShapeInfo();
                    yTad.createOffsets();

                    /**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */
                    int largerElementWiseStride;
                    int smallerElementWiseStride;
                    auto xElementWiseStride = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
                    auto yElementWiseStride = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
                    int tadLength;
                    Nd4jLong xModLength;
                    Nd4jLong yModLength;
                    Nd4jLong *iterationTadInfo;
                    bool xTadBigger;
                    if(shape::length(xShapeInfo) > shape::length(yShapeInfo)) {
                        tadLength = shape::length(xTad.tadOnlyShapeInfo);
                        iterationTadInfo = xTad.tadOnlyShapeInfo;
                        largerElementWiseStride = shape::elementWiseStride(xShapeInfo);
                        smallerElementWiseStride = shape::elementWiseStride(yShapeInfo);
                        xModLength = 1;
                        yModLength = tadLength;
                        xTadBigger = true;

                    }
                    else {
                        tadLength = shape::length(yTad.tadOnlyShapeInfo);
                        iterationTadInfo = yTad.tadOnlyShapeInfo;
                        largerElementWiseStride = shape::elementWiseStride(yShapeInfo);
                        smallerElementWiseStride = shape::elementWiseStride(xShapeInfo);
                        xModLength = tadLength;
                        yModLength = 1;
                        xTadBigger = false;
                    }




                    if (largerElementWiseStride >= 1 && smallerElementWiseStride >= 1 && xElementWiseStride >= 1 && yElementWiseStride >= 1) {
                        if(shape::length(xShapeInfo) == shape::length(yShapeInfo)) {
                            //#pragma omp parallel for proc_bind(AFFINITY) default(shared)
                            for (Nd4jLong i = 0; i < resultLength; i++) {
                                T *localExtraParams = nullptr;
                                if (OpType::extraParamsLen > 0)
                                    localExtraParams = new T[OpType::extraParamsLen];
                                for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                                    localExtraParams[extraParamsIdx] = startingVal;
                                }

                                Nd4jLong offset = xTad.tadOffsets[i];
                                Nd4jLong yOffset = yTad.tadOffsets[i];
                                result[i] = OpType::op(x[offset], y[yOffset], localExtraParams);
                                for (int j = 1; j < tadLength; j++) {
                                    int xIdx = (offset + xElementWiseStride * j);
                                    int yIdx = (yOffset + yElementWiseStride * j);
                                    result[i] = OpType::update(result[i], OpType::op(x[xIdx],
                                                                                     y[yIdx],
                                                                                     localExtraParams), localExtraParams);
                                }

                                result[i] = OpType::postProcess(result[i], tadLength, localExtraParams);

                                if (localExtraParams != nullptr)
                                    delete[] localExtraParams;
                            }
                        }
                        else {
                            int tadsPerThread = resultLength / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());


//#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                            for (int i = 0; i < resultLength; i++) {
                                Nd4jLong xOffset = xTadBigger ? xTad.tadOffsets[i] : 0;
                                Nd4jLong yOffset = !xTadBigger ? yTad.tadOffsets[i] : 0;
                                auto xShape = xTadBigger ? xTad.tadShape : shape::shapeOf(xShapeInfo);
                                auto yShape = !xTadBigger ? yTad.tadShape : shape::shapeOf(yShapeInfo);
                                auto xStride = xTadBigger ? xTad.tadStride : shape::stride(xShapeInfo);
                                auto yStride = !xTadBigger ? yTad.tadStride : shape::stride(yShapeInfo);
                                int xRank = xTadBigger ? shape::rank(xTad.tadOnlyShapeInfo) : shape::rank(xShapeInfo);
                                int yRank = !xTadBigger ? shape::rank(yTad.tadOnlyShapeInfo) : shape::rank(yShapeInfo);
                                Nd4jLong coord[MAX_RANK];
                                Nd4jLong yCoord[MAX_RANK];
                                T start = 0.0;

                                for (int j = 0; j < tadLength; j++) {
                                    if(xTadBigger) {
                                        shape::ind2subC(shape::rank(xTad.tadOnlyShapeInfo),
                                                        xTad.tadStride, j, coord);
                                        shape::ind2subC(shape::rank(yShapeInfo),
                                                        shape::shapeOf(yShapeInfo), j, yCoord);
                                    }
                                    else {
                                        shape::ind2subC(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), j, coord);
                                        shape::ind2subC(shape::rank(yTad.tadOnlyShapeInfo),
                                                        yTad.tadShape, j, yCoord);
                                    }



                                    int xOffset2 =  shape::getOffset(xOffset,xShape,xStride,coord,xRank);
                                    int yOffset2 =  shape::getOffset(yOffset,yShape,yStride,yCoord,yRank);
                                    start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParams), extraParamsVals);
                                }

                                result[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                            }
                        }

                    } else {
                        shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                        xTad.createTadOnlyShapeInfo();
                        xTad.createOffsets();


                        shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                        yTad.createTadOnlyShapeInfo();
                        yTad.createOffsets();
                        int tadsPerThread = resultLength / TAD_THRESHOLD;
                        int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                        Nd4jLong coord[MAX_RANK];

//#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(coord)
                        for (int i = 0; i < resultLength; i++) {
                            Nd4jLong xOffset = xTad.tadOffsets[i];
                            Nd4jLong yOffset = yTad.tadOffsets[i];


                            T start = OpType::startingValue(x + xOffset);

                            for (int j = 0; j < tadLength; j++) {
                                shape::ind2subC(shape::rank(iterationTadInfo), shape::shapeOf(iterationTadInfo), j, coord);
                                Nd4jLong xOffset2 = shape::getOffset(xOffset,shape::shapeOf(xTad.tadOnlyShapeInfo),shape::stride(xTad.tadOnlyShapeInfo),coord,shape::rank(xTad.tadOnlyShapeInfo));
                                Nd4jLong yOffset2 = shape::getOffset(yOffset,shape::shapeOf(yTad.tadOnlyShapeInfo),shape::stride(yTad.tadOnlyShapeInfo),coord,shape::rank(yTad.tadOnlyShapeInfo));
                                start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParamsVals), extraParamsVals);
                            }

                            result[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                        }
                    }


                }
            }
        };
    }
}




#endif /* REDUCE3_H_ */
