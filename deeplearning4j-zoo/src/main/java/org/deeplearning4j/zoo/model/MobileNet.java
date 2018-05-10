package org.deeplearning4j.zoo.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * MobileNet
 *
 * An implementation of MobileNet in Deeplearning4j. MobileNets
 * are based on a streamlined architecture that uses depthwise
 * separable convolutions to build light weight deep
 * neural networks.
 *
 * <p>Paper: https://arxiv.org/abs/1704.04861</p>
 * <p>Weights are available and have been converted from https://keras.io/applications/#mobilenet</p>
 *
 * @author Justin Long (crockpotveggies)
 *
 */
@AllArgsConstructor
@Builder
public class MobileNet extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 512, 512};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private WeightInit weightInit = WeightInit.RELU;
    @Builder.Default private Distribution weightDistribution = new TruncatedNormalDistribution(0, 0.5); // use when weightInit is DISTRIBUTION
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    // MobileNet specific config
    @Builder.Default private double alpha = 1.0;
    @Builder.Default private int depthMultiplier = 1;

    private MobileNet() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return "http://blob.deeplearning4j.org/models/mobilenet_dl4j_imagenet_inference.v1.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.SEGMENT)
            return 111L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(weightInit)
                .dist(new TruncatedNormalDistribution(0.0, 0.5))
                .l2(5e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .graphBuilder();


        graph
                // TODO
                .setOutputs("output").backprop(true).pretrain(false);

        return graph;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}
