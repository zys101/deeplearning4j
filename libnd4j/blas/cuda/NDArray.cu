//
// CUDA-compatible NDArray implementation
//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include "../NDArray.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <pointercast.h>

#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>

namespace nd4j {


////////////////////////////////////////////////////////////////////////
template<typename T>
void* NDArray<T>::operator new(size_t i) {
	if (nd4j::memory::MemoryRegistrator::getInstance()->hasWorkspaceAttached()) {
		nd4j::memory::Workspace* ws = nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace();

		return ws->allocateBytes((Nd4jLong) i);
	} else {
		Nd4jPointer pointer = nullptr;
		auto res = cudaHostAlloc(reinterpret_cast<void **>(&pointer), i, cudaHostAllocDefault);

		if (res != 0)
			throw std::runtime_error("CUDA Host allocation failed!");

		return pointer;
	}
}


////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::operator delete(void* p) {
	if (p != nullptr)
		cudaFreeHost(reinterpret_cast<void *>(p));
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::getView() {

	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template <typename T>
template <typename N>
NDArray<N>* NDArray<T>::asT() {	

        return new NDArray<N>();
}

////////////////////////////////////////////////////////////////////////
// default constructor, do not allocate memory, memory for array is passed from outside 
template <typename T>
NDArray<T>::NDArray(T *buffer, Nd4jLong *shapeInfo, nd4j::memory::Workspace* workspace) 
    :
    _buffer(buffer),
    _shapeInfo(shapeInfo),
    _isBuffAlloc(false), // buffer is a weak link
    _isShapeAlloc(false), // shapeInfo is a weak link
    _workspace(workspace)
{}

////////////////////////////////////////////////////////////////////////
//constructor, create empty array at given workspace
template <typename T>
NDArray<T>::NDArray(nd4j::memory::Workspace* workspace)
    :
    _buffer(nullptr),
    _shapeInfo(nullptr),
    _isBuffAlloc(false),
    _isShapeAlloc(false),
    _workspace(workspace)
{}


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>::NDArray(T scalar) {

}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
template <typename T>
NDArray<T>::NDArray(const Nd4jLong* shapeInfo, const bool copyStrides, nd4j::memory::Workspace* workspace) {

	const int rank = shapeInfo[0];

	if (rank > MAX_RANK)
		throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");	

	_workspace = workspace;

	const auto bufLen = shape::length(const_cast<Nd4jLong*>(shapeInfo));
    const auto shapeLen = shape::shapeInfoLength(const_cast<Nd4jLong*>(shapeInfo));
    
    ALLOCATE(_shapeInfo, _workspace, shapeLen, Nd4jLong);
    ALLOCATE_SPECIAL(_shapeInfoD, _workspace, shapeLen, Nd4jLong);
    ALLOCATE(_buffer, _workspace, bufLen, T);
	ALLOCATE_SPECIAL(_bufferD, _workspace, bufLen, T);
    
    memcpy(_shapeInfo, shapeInfo, shape::shapeInfoByteLength(const_cast<Nd4jLong*>(shapeInfo)));     // copy shape information into new array
    if(!copyStrides)
        shape::updateStrides(_shapeInfo, ordering());
    cudaMemcpy(_shapeInfoD, _shapeInfo, shapeLen * sizeof(T), cudaMemcpyHostToDevice);

    memset(_buffer, 0, bufLen*sizeOfT());          // set all elements in new array to be zeros
    
    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>::NDArray(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {

	const int rank = (int) shape.size();

	if (rank > MAX_RANK)
		throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");	

	_workspace = workspace;

	// allocate host & device buffers for shape
	const auto shapeLen = shape::shapeInfoLength(rank);
	ALLOCATE(_shapeInfo, _workspace, shapeLen, Nd4jLong);
	ALLOCATE_SPECIAL(_shapeInfoD, _workspace, shapeLen, Nd4jLong);

	// fill host _shapeInfo
	int i = 1;
    _shapeInfo[0] = rank;
    for (auto &item: shape)
            _shapeInfo[i++] = item;
    shape::updateStrides(_shapeInfo, order);

    // copy information from host _shapeInfo to device _shapeInfoD, _shapeInfoD already points on pinned-host-memory
    cudaMemcpy(_shapeInfoD, _shapeInfo, shapeLen * sizeof(T), cudaMemcpyHostToDevice);
	
    // allocate host & device buffers for data
    const auto bufLen = shape::length(_shapeInfo);	
	ALLOCATE(_buffer, _workspace, bufLen, T);
	ALLOCATE_SPECIAL(_bufferD, _workspace, bufLen, T);

	if (!data.empty()) {
		/**
		 * FIXME: obviously we dont heed to udpate BOTH buffers here.
		 * But since that's an example, and we don't have dirty tracking yet - let it be for now
		 *
		 * FIXME: once we update signatures, we will be using LaunchContext::stream() and LaunchContext::specialStream()
		 * and we'll be using cudaMemcpyAsync instead
		 */
		// first we copy data to pinned memory, here _buffer already points on pinned-host-memory
		memcpy(_buffer, data.data(), data.size() * sizeof(T));

		// then we copy data to device buffer
		cudaMemcpy(_bufferD, _buffer, bufLen * sizeof(T), cudaMemcpyHostToDevice);
	} 
	else {
		// memset should be applied only on one side, and other one should be tagged as dirty
		memset(_buffer, 0, bufLen * sizeof(T));
	}

	_isBuffAlloc = true;
	_isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>::NDArray(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) 
	:
	NDArray(order, shape, std::vector<T>(), workspace)
{}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>::NDArray(T* buffer, const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {

	const int rank = (int) shape.size();

    if (rank > MAX_RANK)
        throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");

    _workspace = workspace;        
    _buffer = buffer;

    const auto shapeLen = shape::shapeInfoLength(rank);
	ALLOCATE(_shapeInfo, _workspace, shapeLen, Nd4jLong);
	ALLOCATE_SPECIAL(_shapeInfoD, _workspace, shapeLen, Nd4jLong);

    int i = 1;
    _shapeInfo[0] = rank;
    for (auto &item: shape)
            _shapeInfo[i++] = item;
    shape::updateStrides(_shapeInfo, order);

    // copy information from host _shapeInfo to device _shapeInfoD, _shapeInfoD already points on pinned-host-memory
    cudaMemcpy(_shapeInfoD, _shapeInfo, shapeLen * sizeof(T), cudaMemcpyHostToDevice);

    ALLOCATE_SPECIAL(_bufferD, _workspace, shape::length(_shapeInfo), T);

    // copy data from host _buffer to device _bufferD
	cudaMemcpy(_bufferD, _buffer, shape::length(_shapeInfo) * sizeof(T), cudaMemcpyHostToDevice);

    _isBuffAlloc = false;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>::NDArray(const NDArray<T> *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {

	_workspace = workspace;

	const auto shapeLen = shape::shapeInfoLength(other->_shapeInfo[0]);
	ALLOCATE(_shapeInfo, _workspace, shapeLen, Nd4jLong);
	ALLOCATE_SPECIAL(_shapeInfoD, _workspace, shapeLen, Nd4jLong);
	memcpy(_shapeInfo, other->_shapeInfo, shape::shapeInfoByteLength(other->_shapeInfo));
	if(!copyStrides) 
        shape::updateStrides(_shapeInfo, ordering());
	cudaMemcpy(_shapeInfoD, _shapeInfo, shapeLen * sizeof(T), cudaMemcpyHostToDevice);

    const auto bufLen = shape::length(other->_shapeInfo);	
    ALLOCATE(_buffer, _workspace, bufLen, T);
    ALLOCATE_SPECIAL(_bufferD, _workspace, bufLen, T); 

    _isBuffAlloc = true;
    _isShapeAlloc = true;
}

////////////////////////////////////////////////////////////////////////
// copy constructor
template <typename T>
NDArray<T>::NDArray(const NDArray<T>& other) {

	_workspace = other._workspace;

	const int shapeLen = shape::shapeInfoLength(other._shapeInfo[0]);
    ALLOCATE(_shapeInfo, _workspace, shapeLen, Nd4jLong);
    ALLOCATE_SPECIAL(_shapeInfoD, _workspace, shapeLen, Nd4jLong);
    ALLOCATE(_buffer, _workspace, shape::length(other._shapeInfo), T);
    ALLOCATE_SPECIAL(_bufferD, _workspace, shape::length(other._shapeInfo), T); 

    memcpy(_shapeInfo, other._shapeInfo, shape::shapeInfoByteLength(other._shapeInfo));     // copy shape information into new array
    shape::updateStrides(_shapeInfo, other.ordering());
    cudaMemcpy(_shapeInfoD, _shapeInfo, shapeLen * sizeof(T), cudaMemcpyHostToDevice);

    this->assign(&other);
    cudaMemcpy(_bufferD, _buffer, shape::length(_shapeInfo) * sizeof(T), cudaMemcpyHostToDevice);
    
    _isBuffAlloc = true; 
    _isShapeAlloc = true;    
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::string NDArray<T>::toStringValue(T value) {

	return "";
}

////////////////////////////////////////////////////////////////////////
template<>
std::string NDArray<float16>::toStringValue(float16 value) {

	return "";
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::string NDArray<T>::asIndexedString(Nd4jLong limit) {

	return "";
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::string NDArray<T>::asString(Nd4jLong limit) {

	return "";
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::vector<T> NDArray<T>::getBufferAsVector() {

	return std::vector<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::vector<Nd4jLong> NDArray<T>::getShapeAsVector() {

 	return std::vector<Nd4jLong >();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::vector<int64_t> NDArray<T>::getShapeInfoAsFlatVector() {

 	return std::vector<int64_t>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
std::vector<Nd4jLong> NDArray<T>::getShapeInfoAsVector() {

	return std::vector<Nd4jLong>();
}

////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__
template<typename T>
void NDArray<T>::applyTriplewiseLambda(NDArray<T>* second, NDArray<T> *third, const std::function<T(T, T, T)>& func, NDArray<T>* target) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::applyPairwiseLambda(NDArray<T>* other, const std::function<T(T, T)>& func, NDArray<T>* target) {
        
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::applyLambda(const std::function<T(T)>& func, NDArray<T>* target) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray<T>* target) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::applyIndexedPairwiseLambda(NDArray<T>* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray<T>* target) {

}
#endif

////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<int8_t> NDArray<T>::asByteVector() {

	return std::vector<int8_t>();
}

////////////////////////////////////////////////////////////////////////
// move constructor
template <typename T>
NDArray<T>::NDArray(NDArray<T>&& other) noexcept {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
T* NDArray<T>::getBuffer() {
	return new T();
}

template<typename T>
T* NDArray<T>::buffer() {
	return new T();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong* NDArray<T>::getShapeInfo() const{
	return new Nd4jLong();
}

template<typename T>
Nd4jLong* NDArray<T>::shapeInfo() {
	return new Nd4jLong();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
T* NDArray<T>::specialBuffer() {
	return new T();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong* NDArray<T>::specialShapeInfo() {

	return new Nd4jLong();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::setSpecialBuffers(T * buffer, Nd4jLong *shape) {

}

////////////////////////////////////////////////////////////////////////
// assignment operator
template<typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {

	return *this;
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
template <typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T>&& other) noexcept {

    return *this;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>& NDArray<T>::operator=(const T scalar) {
    
    return *this;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::replacePointers(T *buffer, Nd4jLong *shapeInfo, const bool releaseExisting ) {

}

////////////////////////////////////////////////////////////////////////
// This method assigns values of given NDArray to this one, wrt order
template<typename T>
void NDArray<T>::assign(const NDArray<T> *other) {

}

////////////////////////////////////////////////////////////////////////
// This method assigns values of given NDArray to this one
template<typename T>
void NDArray<T>::assign(const NDArray<T>& other) {

}

////////////////////////////////////////////////////////////////////////
// This method assigns given value to all elements in this NDArray
template<typename T>
void NDArray<T>::assign(const T value) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::detach() {

	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
template <typename T>
NDArray<T>* NDArray<T>::dup(const char newOrder) {

    return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
T NDArray<T>::varianceNumber(bool biasCorrected) {

	return T();
}

//////////////////////////////////////////////////////////////////////////
// This method returns sum of all elements of this NDArray
template<typename T>
T NDArray<T>::sumNumber() const {
	return T();
}

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
template<typename T>
T NDArray<T>::meanNumber() const {
    return T();
}

//////////////////////////////////////////////////////////////////////////
// method calculates sum along dimension(s) in this array and save it to row: as new NDArray with dimensions 1xN
template<typename T>
NDArray<T>* NDArray<T>::sum(const std::vector<int> &dimensions) const {

	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
bool NDArray<T>::isContiguous() {

	return true;
}

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
template<typename T>
template<typename OpName>
NDArray<T> *NDArray<T>::reduceAlongDimension(const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
            
	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
template<typename T>
template<typename OpName>
NDArray<T> NDArray<T>::reduceAlongDims(const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
                
    return NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
template<typename T>
template<typename OpName>
void NDArray<T>::reduceAlongDimension(NDArray<T>* target, const std::vector<int>& dimensions, const bool keepDims, const bool supportOldShapes, T *extras) const {

}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::reduceAlongDimension(const std::initializer_list<int>& dimensions, const bool keepDims, const bool supportOldShapes) const {
		        
    return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
T NDArray<T>::reduceNumber(T *extraParams) const {

	return T();
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
Nd4jLong NDArray<T>::indexReduceNumber(T *extraParams) {

	return Nd4jLong();
}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
template<typename T>
template<typename OpName>
void NDArray<T>::applyTransform(NDArray<T> *target, T *extraParams) {

}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
template<typename T>
template<typename OpName>
void NDArray<T>::applyTransform(T *extraParams) {

}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
template<typename T>
template<typename OpName>
NDArray<T> NDArray<T>::transform(T *extraParams) {
    
	return NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// perform pairwise transformation
template<typename T>
template<typename OpName>
void NDArray<T>::applyPairwiseTransform(NDArray<T> *other, T *extraParams) {

}

//////////////////////////////////////////////////////////////////////////
// perform pairwise transformation
template<typename T>
template<typename OpName>
void NDArray<T>::applyPairwiseTransform(NDArray<T> *other, NDArray<T> *target, T *extraParams) {
                                       
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyRandom(nd4j::random::RandomBuffer *buffer, NDArray<T>* y, NDArray<T>* z, T* extraArgs) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
Nd4jLong NDArray<T>::tensorsAlongDimension(std::initializer_list<int> dimensions) const {

	return Nd4jLong();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
Nd4jLong NDArray<T>::tensorsAlongDimension(const std::vector<int>& dimensions) const {
        
	return Nd4jLong();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::tensorAlongDimension(Nd4jLong index, const std::initializer_list<int>& dimensions) const {

	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::printShapeInfo(const char * msg) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::printBuffer(const char* msg, Nd4jLong limit) {
        
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::printIndexedBuffer(const char* msg, Nd4jLong limit) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::tensorAlongDimension(Nd4jLong index, const std::vector<int>& dimensions) const {

	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected 
template <typename T>
NDArray<T>* NDArray<T>::transpose() const {
	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected 
template <typename T>
void NDArray<T>::transpose(NDArray<T>& target) const {

}

////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed 
template <typename T>
void NDArray<T>::transposei() {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
bool NDArray<T>::equalsTo(NDArray<T> &other, T eps) const {
    
    return true;
}

////////////////////////////////////////////////////////////////////////
// This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
template<typename T>
bool NDArray<T>::equalsTo(const NDArray<T> *other, T eps) const {

	return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::addRowVector(const NDArray<T> *row, NDArray<T>* target) const {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::subRowVector(const NDArray<T> *row, NDArray<T>* target) const {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::mulRowVector(const NDArray<T> *row, NDArray<T>* target) const {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::divRowVector(const NDArray<T> *row, NDArray<T>* target) const {

}

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
template<typename T>
void NDArray<T>::addiRowVector(const NDArray<T> *row) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::addColumnVector(const NDArray<T> *column, NDArray<T>* target) const {

}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
template<typename T>
void NDArray<T>::addiColumnVector(const NDArray<T> *column) {

}

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
template<typename T>
void NDArray<T>::muliColumnVector(const NDArray<T> *column) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyScalar(T scalar, NDArray<T>* target, T *extraParams) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyScalar(NDArray<T>& scalar, NDArray<T>* target, T *extraParams) {

}

//////////////////////////////////////////////////////////////////////////
// calculate strides 
template <typename T>
void NDArray<T>::updateStrides(const char order) {
	
}

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T>
bool NDArray<T>::reshapei(const char order, const std::initializer_list<Nd4jLong>& shape) {
	return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::reshapei(const std::initializer_list<Nd4jLong>& shape) {
    return true;
}	

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::reshapei(const std::vector<Nd4jLong>& shape) {
    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::enforce(const std::initializer_list<Nd4jLong> &dimensions, char order) {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::enforce(std::vector<Nd4jLong> &dimensions, char o) {

}

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length 
template <typename T>
bool NDArray<T>::reshapei(const char order, const std::vector<Nd4jLong>& cshape) {

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
Nd4jLong NDArray<T>::argMax(std::initializer_list<int> dimensions) {

	return Nd4jLong();
}

//////////////////////////////////////////////////////////////////////////
// create new array with corresponding order and shape, new array will point to the same _buffer as this array
template <typename T>
NDArray<T>* NDArray<T>::reshape(const char order, const std::vector<Nd4jLong>& shape) const {

	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
void NDArray<T>::tilei(const std::vector<Nd4jLong>& reps) {
	
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
NDArray<T> NDArray<T>::tile(const std::vector<Nd4jLong>& reps) const {
    
    return NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
template <typename T>
void NDArray<T>::tile(const std::vector<Nd4jLong>& reps, NDArray<T>& target) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::tile(NDArray<T>& target) const {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
Nd4jLong NDArray<T>::sizeAt(int dim) const {

	return Nd4jLong();
}

//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
template<typename T>
NDArray<T>* NDArray<T>::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {

    return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
template<typename T>
void NDArray<T>::repeat(int dimension, NDArray<T>& target) const {

}


//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const int* dimensions, const int rank) {

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const Nd4jLong* dimensions, const int rank) {

	return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::initializer_list<int>& dimensions) {
    
    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::vector<int>& dimensions) {
    
    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::initializer_list<Nd4jLong>& dimensions) {

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool NDArray<T>::permutei(const std::vector<Nd4jLong>& dimensions) {

    return true;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const int* dimensions, const int rank) const {

    return new NDArray<T>();
}

/////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const Nd4jLong* dimensions, const int rank) const {

	return new NDArray<T>();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::vector<int>& dimensions) const {

	return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::vector<Nd4jLong>& dimensions) const {

	return new NDArray<T>();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::initializer_list<int>& dimensions) const {
    
    std::vector<int> vec(dimensions);
    return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::permute(const std::initializer_list<Nd4jLong>& dimensions) const {
    
    std::vector<Nd4jLong> vec(dimensions);
		return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const int* dimensions, const int rank, NDArray<T>& target) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const Nd4jLong *dimensions, const int rank, NDArray<T>& target) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const std::vector<int>& dimensions, NDArray<T>& target) const {

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::permute(const std::vector<Nd4jLong>& dimensions, NDArray<T>& target) const {
	
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyBroadcast(std::initializer_list<int> dimensions, const NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyBroadcast(std::vector<int>& dimensions, const NDArray<T>* tadArray, NDArray<T>* target, T* extraArgs) {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
void NDArray<T>::applyTrueBroadcast(const NDArray<T>* other, NDArray<T>* target, const bool checkTargetShape, T *extraArgs) const {

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T>* NDArray<T>::applyTrueBroadcast(const NDArray<T>* other, T *extraArgs) const {
 
    return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
template <typename OpName>
NDArray<T> NDArray<T>::applyTrueBroadcast(const NDArray<T>& other, T *extraArgs) const {

    return NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// return array which is broadcasted from this and argument array  
template<typename T>
NDArray<T>* NDArray<T>::broadcast(const NDArray<T>& other) {	

    return new NDArray<T>();
}

//////////////////////////////////////////////////////////////////////////
// check whether array's rows (arg=0) or columns (arg=1) create orthogonal basis
template<typename T>
bool NDArray<T>::hasOrthonormalBasis(const int arg) {
    
    return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is identity matrix
template<typename T>
bool NDArray<T>::isIdentityMatrix() {

	return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is unitary matrix
template<typename T>
bool NDArray<T>::isUnitary() {
    
    return true;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::subarray(IndicesList& idx, std::vector<Nd4jLong>& strides) const {

    return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::subarray(IndicesList& idx) const {
    
    return new NDArray<T>();
}
    
////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::subarray(const std::initializer_list<NDIndex*>& idx) const {

    return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::subarray(const Intervals& idx) const {

    return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArray<T>::cast(DataType dtype) {

	return nullptr;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray<T>::cast(NDArray<T>* target, DataType dtype) {
	
}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::applyIndexReduce(const NDArray<T>* target, const std::vector<int>& dimensions, const T *extraParams) const {

}

////////////////////////////////////////////////////////////////////////
// reduce dimensions in this array relying on index operations
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::applyIndexReduce(const std::vector<int>& dimensions, const T* extraParams ) const {
        
	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 operations to this and other array, return result in new output array
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const T* extraParams) const {
	
	return new NDArray<T>();
}
    
////////////////////////////////////////////////////////////////////////
// apply reduce3 (execAll) operations to this and other array, return result in new output array
template<typename T>
template<typename OpName>
NDArray<T>*  NDArray<T>::applyAllReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {

	return new NDArray<T>();
}
 
////////////////////////////////////////////////////////////////////////
// apply reduce3 (exec) operations to this and other array, return result in new output array
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::applyReduce3(const NDArray<T>* other, const std::vector<int>& dimensions, const T* extraParams) const {

	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::vector<int>& dimensions) const {

	return new NDArray<T>();    
}
    
////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
NDArray<T>* NDArray<T>::varianceAlongDimension(const bool biasCorrected, const std::initializer_list<int>& dimensions) const {
    
	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::vector<int>& dimensions) {
	
}

////////////////////////////////////////////////////////////////////////
template<typename T>
template<typename OpName>
void NDArray<T>::varianceAlongDimension(const NDArray<T> *target, const bool biasCorrected, const std::initializer_list<int>& dimensions) {

}

////////////////////////////////////////////////////////////////////////
// operator returns sub-array with buffer pointing at this->_buffer + certain offset
template<typename T>
NDArray<T> NDArray<T>::operator()(const int* idx, bool keepUnitiesInShape)  const {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// operator returns sub-array with buffer pointing at this->_buffer + certain offset
template<typename T>
NDArray<T> NDArray<T>::operator()(const Intervals& idx, bool keepUnitiesInShape)  const {

	return NDArray<T>();
}
        
////////////////////////////////////////////////////////////////////////
// addition operator array + array
template<typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& other) const {

	return NDArray<T>();

}

////////////////////////////////////////////////////////////////////////
// addition operator array + scalar
template<typename T>
NDArray<T> NDArray<T>::operator+(const T scalar) const {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// addition operator scalar + array
// template<typename T>
// NDArray<T> operator+(const T scalar, const NDArray<T>& arr) {
//     return arr + scalar;
// }
ND4J_EXPORT NDArray<float16> operator+(const float16 scalar, const NDArray<float16>& arr) {
	return NDArray<float16>();
}
ND4J_EXPORT NDArray<float> operator+(const float scalar, const NDArray<float>& arr) {
    return NDArray<float>();
}
ND4J_EXPORT NDArray<double> operator+(const double scalar, const NDArray<double>& arr) {
	return NDArray<double>();        
}

////////////////////////////////////////////////////////////////////////
// subtraction operator scalar - array
// template<typename T>
// NDArray<T> operator-(const T scalar, const NDArray<T>& arr) {

//     NDArray<T> result(arr._shapeInfo, false, arr._workspace);
//     functions::scalar::ScalarTransform<T>::template transform<simdOps::ReverseSubtract<T>>(arr._buffer, arr._shapeInfo, result._buffer, result._shapeInfo, scalar, nullptr);

//     return result;
// }    
ND4J_EXPORT NDArray<float16> operator-(const float16 scalar, const NDArray<float16>& arr) {
	return NDArray<float16>();
}        
ND4J_EXPORT NDArray<float> operator-(const float scalar, const NDArray<float>& arr) {
	return NDArray<float>();
}        
ND4J_EXPORT NDArray<double> operator-(const double scalar, const NDArray<double>& arr) {
	return NDArray<double>();
}    
    
////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::operator+=(const NDArray<T>& other) {    

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::operator-=(const NDArray<T>& other) {    

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::operator+=(const T other) {

}

////////////////////////////////////////////////////////////////////////    
template<typename T>
void NDArray<T>::operator-=(const T other) {  

}

////////////////////////////////////////////////////////////////////////
// subtraction operator array - array
template<typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T>& other) const {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// subtraction operator array - scalar
template<typename T>
NDArray<T> NDArray<T>::operator-(const T& scalar) const {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// negative operator, it makes all array elements = -elements
template<typename T>
NDArray<T> NDArray<T>::operator-() const {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// multiplication operator array*array
template<typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T>& other) const {
        
	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// multiplication operator array*scalar
template<typename T>
NDArray<T> NDArray<T>::operator*(const T scalar) const {
        
	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
// multiplication operator array1 *= array2
template<typename T>
void NDArray<T>::operator*=(const NDArray<T>& other) {    

}

////////////////////////////////////////////////////////////////////////
// multiplication operator array*scalar
template<typename T>
void NDArray<T>::operator*=(const T scalar) {

}


////////////////////////////////////////////////////////////////////////
// division operator array/array
template<typename T>
NDArray<T> NDArray<T>::operator/(const NDArray<T>& other) const {
	NDArray<T> cs;
	return cs;
}

////////////////////////////////////////////////////////////////////////
// division operator array / scalar
template<typename T>
NDArray<T> NDArray<T>::operator/(const T scalar) const {
	NDArray<T> cs;
	return cs;
}

////////////////////////////////////////////////////////////////////////
// division operator array1 /= array2
template<typename T>
void NDArray<T>::operator/=(const NDArray<T>& other) {

}

////////////////////////////////////////////////////////////////////////
// division operator array /= scalar
template<typename T>
void NDArray<T>::operator/=(const T scalar) {
        	
}

////////////////////////////////////////////////////////////////////////
// mathematical multiplication of two arrays
template<typename T>
NDArray<T> mmul(const NDArray<T>& left, const NDArray<T>& right) {

	return NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
DataType NDArray<T>::dataType() const {
	
	return DataType_INHERIT;
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::assign(const NDArray<T>& other, const Intervals& idx) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::setIdentity() {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::swapUnsafe(NDArray<T>& other) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray<T>* NDArray<T>::diagonal(const char type) const {        
	
	return new NDArray<T>();
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::setValueInDiagMatrix(const T& value, const int diag, const char direction) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::streamline(char o) {

}


////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::tileToShape(const std::vector<Nd4jLong>& shape, NDArray<T>* target) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray<T>::tileToShape(const std::initializer_list<Nd4jLong>& shape, NDArray<T>* target) {

}

////////////////////////////////////////////////////////////////////////
template<typename T>
T NDArray<T>::getTrace() const {
    
    return T();
}
////////////////////////////////////////////////////////////////////////
// default destructor
template<typename T>
NDArray<T>::~NDArray() noexcept {
	// TODO: we probably want streamSync here, based on LaunchContext

	if (_isBuffAlloc && _workspace == nullptr && _buffer != nullptr) {
		cudaFreeHost(reinterpret_cast<void *>(_buffer));
		cudaFree(reinterpret_cast<void *>(_bufferD));
	}

	if (_isShapeAlloc  && _workspace == nullptr && _shapeInfo != nullptr) {
		cudaFreeHost(reinterpret_cast<void *>(_shapeInfo));
		cudaFree(reinterpret_cast<void *>(_shapeInfoD));
	}
}

	template class ND4J_EXPORT NDArray<float>;
	template class ND4J_EXPORT NDArray<float16>;
	template class ND4J_EXPORT NDArray<double>;


	template NDArray<float>* NDArray<float>::asT<float>();
	template NDArray<float16>* NDArray<float>::asT<float16>();
	template NDArray<double>* NDArray<float>::asT<double>();

	template NDArray<float>* NDArray<float16>::asT<float>();
	template NDArray<float16>* NDArray<float16>::asT<float16>();
	template NDArray<double>* NDArray<float16>::asT<double>();

	template NDArray<float>* NDArray<double>::asT<float>();
	template NDArray<float16>* NDArray<double>::asT<float16>();
	template NDArray<double>* NDArray<double>::asT<double>();


#ifndef __CLION_IDE__
#include "../cpu/NDArray.macro"
#endif
}