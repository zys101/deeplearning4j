#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <NDArrayFactory.h>
#include <numeric>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
void triu(const NDArray<T>& input, NDArray<T>& output, const int diagonal) {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void triuBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI, const int diagonal) {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void trace(const NDArray<T>& input, NDArray<T>& output) {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle(NDArray<T>& input, NDArray<T>& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

}

////////////////////////////////////////////////////////////////////////
// initial values of inIdx, outIdx, dim must be equal to zero
template<typename T>
void recursiveLoopForPad(const int mode, NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx ) {

}


////////////////////////////////////////////////////////////////////////
template<typename T>
void invertPermutation(const NDArray<T>& input, NDArray<T>& output) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void gatherND(NDArray<T>& input, NDArray<T>& indices, NDArray<T>& output) {

}


////////////////////////////////////////////////////////////////////////
template<typename T>
void gather(NDArray<T>* input, const NDArray<T>* indices, NDArray<T>* output, const std::vector<int>& intArgs) {


}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void eye(NDArray<T>& output) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void scatterUpdate(NDArray<T>& operand, NDArray<T>& updates, const std::vector<int>* intArgs) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeMaxIndex(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {

}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeMax(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeAvg(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {

}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeAdd(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByNorm(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByAveraged(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace) {

}

template void triu<float>(const NDArray<float>& input, NDArray<float>& output, const int diagonal);
template void triu<float16>(const NDArray<float16>& input, NDArray<float16>& output, const int diagonal);
template void triu<double>(const NDArray<double>& input, NDArray<double>& output, const int diagonal);

template void triuBP<float>(const NDArray<float>& input, const NDArray<float>& gradO, NDArray<float>& gradI, const int diagonal);
template void triuBP<float16>(const NDArray<float16>& input, const NDArray<float16>& gradO, NDArray<float16>& gradI, const int diagonal);
template void triuBP<double>(const NDArray<double>& input, const NDArray<double>& gradO, NDArray<double>& gradI, const int diagonal);

template void trace<float>(const NDArray<float>& input, NDArray<float>& output);
template void trace<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void trace<double>(const NDArray<double>& input, NDArray<double>& output);

template void randomShuffle<float>(NDArray<float>& input, NDArray<float>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<float16>(NDArray<float16>& input, NDArray<float16>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<double>(NDArray<double>& input, NDArray<double>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);

template void recursiveLoopForPad<float>(const int mode, NDArray<float>& input, const NDArray<float>& paddings, NDArray<float>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx);
template void recursiveLoopForPad<float16>(const int mode, NDArray<float16>& input, const NDArray<float16>& paddings, NDArray<float16>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx);
template void recursiveLoopForPad<double>(const int mode, NDArray<double>& input, const NDArray<double>& paddings, NDArray<double>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx);

template void invertPermutation<float>(const NDArray<float>& input, NDArray<float>& output);
template void invertPermutation<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void invertPermutation<double>(const NDArray<double>& input, NDArray<double>& output);

template void gatherND<float>(NDArray<float>& input, NDArray<float>& indices, NDArray<float>& output);
template void gatherND<float16>(NDArray<float16>& input, NDArray<float16>& indices, NDArray<float16>& output);
template void gatherND<double>(NDArray<double>& input, NDArray<double>& indices, NDArray<double>& output);

template void gather<float>(NDArray<float>* input, const NDArray<float>* indices, NDArray<float>* output, const std::vector<int>& intArgs);
template void gather<float16>(NDArray<float16>* input, const NDArray<float16>* indices, NDArray<float16>* output, const std::vector<int>& intArgs);
template void gather<double>(NDArray<double>* input, const NDArray<double>* indices, NDArray<double>* output, const std::vector<int>& intArgs);

template void eye<float>(NDArray<float>& output);
template void eye<float16>(NDArray<float16>& output);
template void eye<double>(NDArray<double>& output);

template void scatterUpdate<float>(NDArray<float>& operand, NDArray<float>& updates, const std::vector<int>* intArgs);
template void scatterUpdate<float16>(NDArray<float16>& operand, NDArray<float16>& updates, const std::vector<int>* intArgs);
template void scatterUpdate<double>(NDArray<double>& operand, NDArray<double>& updates, const std::vector<int>* intArgs);

template void mergeMaxIndex<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeMaxIndex<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeMaxIndex<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeMax<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeMax<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeMax<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeAvg<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeAvg<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeAvg<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeAdd<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeAdd<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeAdd<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void clipByNorm<float>(NDArray<float>& input, NDArray<float>& output, const std::vector<int>& dimensions, const float clipNorm, const bool isInplace);
template void clipByNorm<float16>(NDArray<float16>& input, NDArray<float16>& output, const std::vector<int>& dimensions, const float16 clipNorm, const bool isInplace);
template void clipByNorm<double>(NDArray<double>& input, NDArray<double>& output, const std::vector<int>& dimensions, const double clipNorm, const bool isInplace);

template void clipByAveraged<float>(NDArray<float>& input, NDArray<float>& output, const std::vector<int>& dimensions, const float clipNorm, const bool isInplace);
template void clipByAveraged<float16>(NDArray<float16>& input, NDArray<float16>& output, const std::vector<int>& dimensions, const float16 clipNorm, const bool isInplace);
template void clipByAveraged<double>(NDArray<double>& input, NDArray<double>& output, const std::vector<int>& dimensions, const double clipNorm, const bool isInplace);

}
}
}