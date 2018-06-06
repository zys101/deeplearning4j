//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.05.2018
// 

#include "testlayers.h"
#include <NDArray.h>
#include <cuda_runtime_api.h>
#include <memory/MemoryRegistrator.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTestCu : public testing::Test {
public:    
    
};


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test1) {
    
    Nd4jLong fShapeInfo[8] = {2, 2, 2, 1, 2, 0, 1, 102};
    float buffer[4] = {1.,2.,3.,4.};
    
    NDArray<float> arr1(buffer, fShapeInfo);
    NDArray<float> arr2(nd4j::memory::MemoryRegistrator::getInstance()->getWorkspace());
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
  
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);    

    ASSERT_TRUE(bufferD[0] == 1.);
    ASSERT_TRUE(bufferD[1] == 2.);
    ASSERT_TRUE(bufferD[2] == 3.);
    ASSERT_TRUE(bufferD[3] == 4.);

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 2);
    ASSERT_TRUE(shapeInfoD[2] == 2);
    ASSERT_TRUE(shapeInfoD[3] == 1);
    ASSERT_TRUE(shapeInfoD[4] == 2);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 102); 

    ASSERT_TRUE(arr1.lengthOf() == 4);  
    
    delete []shapeInfoD;
    delete []bufferD;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test2) {
    
    NDArray<float> arr1(58.);
        
    Nd4jLong* shapeInfoD = new Nd4jLong[shape::shapeInfoLength(arr1.getShapeInfo())];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shape::shapeInfoByteLength(arr1.getShapeInfo()), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);
            
    ASSERT_TRUE(bufferD[0]  == 58.);

    ASSERT_TRUE(shapeInfoD[0] == 0);
    ASSERT_TRUE(shapeInfoD[1] == 0);
    ASSERT_TRUE(shapeInfoD[2] == 1);
    ASSERT_TRUE(shapeInfoD[3] == 99);    
    
    ASSERT_TRUE(arr1.lengthOf() == 1);     

    delete []shapeInfoD;    
    delete []bufferD;    
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test3) {
    
    Nd4jLong cShapeInfo[8] = {2, 3, 4, 4, 1, 0, 1, 99};
    
    NDArray<float> arr1(cShapeInfo, true);
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
    
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);
            
    ASSERT_TRUE(bufferD[0]  == 0.);
    ASSERT_TRUE(bufferD[2]  == 0.);
    ASSERT_TRUE(bufferD[5]  == 0.);
    ASSERT_TRUE(bufferD[9]  == 0.);
    ASSERT_TRUE(bufferD[11] == 0.);

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 3);
    ASSERT_TRUE(shapeInfoD[2] == 4);
    ASSERT_TRUE(shapeInfoD[3] == 4);
    ASSERT_TRUE(shapeInfoD[4] == 1);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 99); 

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []shapeInfoD;    
    delete []bufferD;    
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test4) {    
    
    NDArray<float> arr1('f', {3,4}, {1.,2.,3,4,5,6,7,8,9,10,11,12});
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
    
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < arr1.lengthOf(); ++i)
        ASSERT_TRUE(bufferD[i]  == i+1.);

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 3);
    ASSERT_TRUE(shapeInfoD[2] == 4);
    ASSERT_TRUE(shapeInfoD[3] == 1);
    ASSERT_TRUE(shapeInfoD[4] == 3);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 102); 

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []shapeInfoD;    
    delete []bufferD;    
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test5) {
    
    NDArray<float> arr1('f', {3,4});
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
    
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < arr1.lengthOf(); ++i)
        ASSERT_TRUE(bufferD[i]  == 0.);

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 3);
    ASSERT_TRUE(shapeInfoD[2] == 4);
    ASSERT_TRUE(shapeInfoD[3] == 1);
    ASSERT_TRUE(shapeInfoD[4] == 3);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 102); 

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []shapeInfoD;    
    delete []bufferD;    
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test6) {    
    
    float buffer[12] = {1.,2.,3,4,5,6,7,8,9,10,11,12};

    NDArray<float> arr1(buffer, 'f', {3,4});
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
    
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < arr1.lengthOf(); ++i)
        ASSERT_TRUE(bufferD[i]  == i+1.);

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 3);
    ASSERT_TRUE(shapeInfoD[2] == 4);
    ASSERT_TRUE(shapeInfoD[3] == 1);
    ASSERT_TRUE(shapeInfoD[4] == 3);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 102); 

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []shapeInfoD;    
    delete []bufferD;    
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test7) {    
    
    NDArray<float> arr2('f', {3,4});
    NDArray<float> arr1(&arr2);
        
    Nd4jLong* shapeInfoD = new Nd4jLong[shape::shapeInfoLength(arr1.getShapeInfo())];
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shape::shapeInfoByteLength(arr1.getShapeInfo()), cudaMemcpyDeviceToHost);    

    ASSERT_TRUE(shapeInfoD[0] == 2);
    ASSERT_TRUE(shapeInfoD[1] == 3);
    ASSERT_TRUE(shapeInfoD[2] == 4);
    ASSERT_TRUE(shapeInfoD[3] == 1);
    ASSERT_TRUE(shapeInfoD[4] == 3);
    ASSERT_TRUE(shapeInfoD[5] == 0);
    ASSERT_TRUE(shapeInfoD[6] == 1);
    ASSERT_TRUE(shapeInfoD[7] == 102); 

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []shapeInfoD;        
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, assign_test1) {    
    
    const float val = 58.;
    NDArray<float> arr1('f', {3,4});
    arr1.assign(val);        

    float* bufferD = new float[arr1.lengthOf()];
    cudaMemcpy(bufferD, arr1.specialBuffer(),  arr1.lengthOf() * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < arr1.lengthOf(); ++i)
        ASSERT_TRUE(bufferD[i]  == val);

    ASSERT_TRUE(arr1.lengthOf() == 12); 
    
    delete []bufferD;
}
