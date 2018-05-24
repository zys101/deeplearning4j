#include <ops/ops.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <NDArrayFactory.h>
#include <ops/declarable/helpers/prefix.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T, typename OpName>
void _prefix(T* x, Nd4jLong* xShapeInfo, T* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse) {

}

///////////////////////////////////////////////////////////////////
template <typename T, typename OpName>
void _prefix(NDArray<T>* x, NDArray<T>* z, std::vector<int>& dims, bool exclusive, bool reverse) {

}

template void _prefix<float, simdOps::Add<float>>(float* x, Nd4jLong* xShapeInfo, float* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
template void _prefix<float16, simdOps::Add<float16>>(float16* x, Nd4jLong* xShapeInfo, float16* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
template void _prefix<double, simdOps::Add<double>>(double* x, Nd4jLong* xShapeInfo, double* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);

template void _prefix<float, simdOps::Multiply<float>>(float* x, Nd4jLong* xShapeInfo, float* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
template void _prefix<float16, simdOps::Multiply<float16>>(float16* x, Nd4jLong* xShapeInfo, float16* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
template void _prefix<double, simdOps::Multiply<double>>(double* x, Nd4jLong* xShapeInfo, double* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);


template void _prefix<float, simdOps::Add<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims, bool exclusive, bool reverse);
template void _prefix<float16, simdOps::Add<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims, bool exclusive, bool reverse);
template void _prefix<double, simdOps::Add<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims, bool exclusive, bool reverse);

template void _prefix<float, simdOps::Multiply<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims, bool exclusive, bool reverse);
template void _prefix<float16, simdOps::Multiply<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims, bool exclusive, bool reverse);
template void _prefix<double, simdOps::Multiply<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims, bool exclusive, bool reverse);

}
}
}