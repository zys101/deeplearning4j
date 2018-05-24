#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <int N, bool B2S>
struct SpaceToBatchHelper {
    
    template <typename T>
    static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
         
    }
};

//////////////////////////////////////////////////////////////////////////
template <bool B2S>
struct SpaceToBatchHelper<0, B2S> {

    template <typename T>
    static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {

    }
};

//////////////////////////////////////////////////////////////////////////
template <typename T, int NUM_BLOCK_DIMS, bool B2S>
void _execute(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
    
}


template void _execute<float, 4, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 3, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 2, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 1, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

template void _execute<float16, 4, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 3, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 2, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 1, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

template void _execute<double, 4, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 3, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 2, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 1, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);


template void _execute<float, 4, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 3, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 2, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float, 1, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

template void _execute<float16, 4, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 3, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 2, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<float16, 1, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

template void _execute<double, 4, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 3, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 2, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
template void _execute<double, 1, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

}
}
}