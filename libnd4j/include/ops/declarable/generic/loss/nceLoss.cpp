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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 26.03.2019
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_nce_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(nce_loss, 4, 1, false, 0, 2) {
  	
  	auto input   = INPUT_VARIABLE(0);		// [bS, dim]
    auto weights = INPUT_VARIABLE(1);		// [numClasses, dim]
    auto biases  = INPUT_VARIABLE(2);		// [numClasses]
    auto lables  = INPUT_VARIABLE(3);		// [bS, numTrue]
    auto output  = OUTPUT_VARIABLE(0);		// [bS]

    const int numSampled = INT_ARG(0);    
    const int numClasses = INT_ARG(1);
    const int numTrue	 = block->getIArguments()->size() > 1 ? INT_ARG(2) : 1;

	// input validation 
    const Nd4jLong bS  = input->sizeAt(0);
    const Nd4jLong dim = input->sizeAt(1);
           		       
    REQUIRE_TRUE(weigths->isSameShape({numClasses, dim}), 0, "NCE_LOSS OP: weights array has wrong shape, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({numClasses, dim}).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    REQUIRE_TRUE(biases ->isSameShape({numClasses}),      0, "NCE_LOSS OP: biases array has wrong shape, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({numClasses}).c_str(), ShapeUtils::shapeAsString(biases).c_str());
    REQUIRE_TRUE(labels ->isSameShape({bS, numTrue}),	  0, "NCE_LOSS OP: labels array has wrong shape, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, numTrue}).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    REQUIRE_TRUE(output ->isSameShape({bS}),	  		  0, "NCE_LOSS OP: output array has wrong shape, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS}).c_str(), ShapeUtils::shapeAsString(output).c_str());

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss) {
	
	getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS})
					->setAllowedInputTypes(1, {ALL_FLOATS})
					->setAllowedInputTypes(2, {ALL_FLOATS})
					->setAllowedInputTypes(3, {ALL_INTS})
					->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(nce_loss) {
	
	auto inputShapeInfo   = inputShape->at(0);
	auto weightsShapeInfo = inputShape->at(1);
	auto biasesShapeInfo  = inputShape->at(2);
    auto labelsShapeInfo  = inputShape->at(3);


    const int numSampled = INT_ARG(0);    
    const int numClasses = INT_ARG(1);
    const int numTrue	 = block->getIArguments()->size() > 1 ? INT_ARG(2) : 1;
	
    const Nd4jLong bS  = input->sizeAt(0);
    const Nd4jLong dim = input->sizeAt(1);

    const std::string expectedWeightsShape = ShapeUtils::shapeAsString({numClasses, dim})
	REQUIRE_TRUE(ShapeUtils::shapeAsString(weightsShapeInfo) == expectedWeightsShape, 0, "NCE_LOSS OP: weights array has wrong shape, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
	const std::string expectedBiasesShape = ShapeUtils::shapeAsString({numClasses})
    REQUIRE_TRUE(ShapeUtils::shapeAsString(biasesShapeInfo) == expectedBiasesShape,   0, "NCE_LOSS OP: biases array has wrong shape, expected is %s, but got %s instead !", expectedBiasesShape.c_str(), ShapeUtils::shapeAsString(biasesShapeInfo).c_str());
    const std::string expectedLablesShape = ShapeUtils::shapeAsString({bS, numTrue})
    REQUIRE_TRUE(ShapeUtils::shapeAsString(labelsShapeInfo) == expectedLablesShape,	  0, "NCE_LOSS OP: labels array has wrong shape, expected is %s, but got %s instead !", expectedLablesShape.c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

	DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(weightsShapeInfo));
	Nd4jLong* outShapeInfo = ShapeBuilders::createVectorShapeInfo(outType, bS, block.getWorkspace());
    
    return SHAPELIST(outShapeInfo);    
}








}
}

#endif