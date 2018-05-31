//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.05.2018
//

#include "testlayers.h"
#include <NDArray.h>
#include <cuda_runtime_api.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTestCu : public testing::Test {
public:    
    
};


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTestCu, constructor_test1) {
    
    Nd4jLong cShapeInfo[8] = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong fShapeInfo[8] = {2, 3, 4, 1, 3, 0, 1, 102};
    float buffer[4] = {1,2,3,4};

    
    NDArray<float> arr1(cShapeInfo, true);
    
    const int rank = arr1.rankOf();
    const int shapeLen = 2 * rank + 4;
    Nd4jLong* shapeInfoD = new Nd4jLong[shapeLen];
        
    cudaMemcpy(shapeInfoD, arr1.specialShapeInfo(), shapeLen * sizeof(Nd4jLong), cudaMemcpyDeviceToHost);    
    // printf("%d \n", (int)arr1.getShapeInfo()[0]); // == 0, why !!??? O_o    
    printf("%lld \n", shapeInfoD[0]); // == 0, why !!??? O_o

    Nd4jLong* rankD = new Nd4jLong();
    cudaMemcpy(rankD, arr1.specialShapeInfo(),sizeof(Nd4jLong), cudaMemcpyHostToDevice);
    printf("%d \n", (int)*rankD);     // == 0, why !!??? O_o   
        
    // ASSERT_TRUE(shapeInfoD[0] == 2); 
    
    // ASSERT_TRUE(shapeInfoD[1] == 3);
    // ASSERT_TRUE(shapeInfoD[3] == 4);
    // ASSERT_TRUE(shapeInfoD[5] == 0); 
    // ASSERT_TRUE(shapeInfoD[7] == 99); 

    // ASSERT_TRUE(arr1.lengthOf() == 2);  // order
    

    // NDArray<float> arr2('f', {2, 2}, {1,2,3,4});
    // NDArray<float> arr3('c', {2, 2});
    // NDArray<float> arr4(buffer, 'c', {2,2}, nullptr);
    // NDArray<float> arr5(&arr4, true);
    // NDArray<float> arr6 = arr5;
    delete []shapeInfoD;
    delete rankD;
}



