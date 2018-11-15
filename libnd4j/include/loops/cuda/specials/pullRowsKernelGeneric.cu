/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//

#include <loops/special_kernels.h>


template <typename T>
__device__ void pullRowsKernelGeneric(void *vx, Nd4jLong *xShapeInfo,
                                     void *vz, Nd4jLong *zShapeInfo,
                                     Nd4jLong len,
                                     Nd4jLong *indexes,
                                     Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                                     Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

	auto x = static_cast<T*>(vx);
	auto z = static_cast<T*>(vz);
    auto xEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(zTadShapeInfo);
    auto tadLength = shape::length(tadShapeInfo);

    if (xEWS >= 1 && zEWS >= 1) {
        for (int idx = blockIdx.x; idx < len; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        }
    } 
    else {
        for (int idx = blockIdx.x; idx < len; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
		    	auto xOffset = shape::getIndexOffset(i, tadShapeInfo, tadLength);
		    	auto zOffset = shape::getIndexOffset(i, zTadShapeInfo, tadLength);
                rZ[zOffset] = rX[xOffset];
            }
        }
    }
}