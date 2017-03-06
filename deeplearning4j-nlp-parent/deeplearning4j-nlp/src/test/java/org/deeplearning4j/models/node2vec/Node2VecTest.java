package org.deeplearning4j.models.node2vec;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.sequencevectors.graph.enums.SamplingMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Edge;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.graph.walkers.impl.NearestVertexWalker;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Node2VecTest {
    private static long seed = 119;

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testPVDM1() throws Exception {
        Graph<VocabWord, Integer> graph = getGraph(100000);
        GraphWalker<VocabWord> walker = new NearestVertexWalker.Builder<VocabWord>(graph)
                .setSamplingMode(SamplingMode.MAX_POPULARITY)
                .setDepth(2)
                .setSeed(119)
                .build();

        Node2Vec<VocabWord, Integer> node2Vec = new Node2Vec.Builder<VocabWord, Integer>(walker)
                .sequenceLearningAlgorithm(new DM<VocabWord>())
                .seed(119)
                .learningRate(0.025)
                .minLearningRate(0.0001)
                .epochs(3)
                .useHierarchicSoftmax(true)
                .negativeSample(0.0)
                .trainElementsRepresentation(true)
                .build();

        node2Vec.fit();

        assertEquals(100001, node2Vec.getVocab().numWords());
        assertEquals(node2Vec.getVocab().tokenFor("0").getElementFrequency(), node2Vec.getVocab().tokenFor("100000").getElementFrequency(), 1e-3);

        double simAZ = node2Vec.similarity("0","100000");
        double simAB = node2Vec.similarity("0","2");

        log.info("0 -> 100000 similarity: {}", simAZ);
        log.info("0 -> 2 similarity: {}", simAB);

//        assertTrue(simAZ > simAB);

        log.info("Nearest: {}",node2Vec.similarity("100000", node2Vec.wordsNearest("100000", 10)));
    }

    @Test
    public void testPVDBOW1() throws Exception {
        Graph<VocabWord, Integer> graph = getGraph(10000);
        GraphWalker<VocabWord> walker = new NearestVertexWalker.Builder<VocabWord>(graph)
                .setSamplingMode(SamplingMode.MAX_POPULARITY)
                .setDepth(2)
                .setSeed(119)
                .build();

        Node2Vec<VocabWord, Integer> node2Vec = new Node2Vec.Builder<VocabWord, Integer>(walker)
                .sequenceLearningAlgorithm(new DBOW<VocabWord>())
                .seed(119)
                .learningRate(0.025)
                .epochs(3)
                .workers(1)
                .trainElementsRepresentation(false)
                .build();

        node2Vec.fit();

        assertEquals(10001, node2Vec.getVocab().numWords());
        assertEquals(node2Vec.getVocab().tokenFor("0").getElementFrequency(), node2Vec.getVocab().tokenFor("10000").getElementFrequency(), 1e-3);

        double simAZ = node2Vec.similarity("0","10000");
        double simAB = node2Vec.similarity("0","2");

        log.info("0 -> 10000 similarity: {}", simAZ);
        log.info("0 -> 2 similarity: {}", simAB);

        assertTrue(simAZ > simAB);
    }

    /**
     * This method returns random generated
     *
     * @return
     */
    protected static Graph<VocabWord, Integer> getGraph(int limitSize) {
        List<Vertex<VocabWord>> list = new ArrayList<>();
        for (int i = 0; i < limitSize; i++) {
            list.add(new Vertex<VocabWord>(i, new VocabWord(1.0, String.valueOf(i))));
        }

        Graph<VocabWord, Integer> graph = new Graph<>(list);
        Random rng = new Random(seed);

        for (int i = 0; i < limitSize; i++) {
            int conns = rng.nextInt( 50) + 10;

            for (int e = 0; e < conns; e++) {
                graph.addEdge(i, getConnection(i, list.size(), rng), 1, true);
            }
        }

        rng = new Random(seed);
        int numConns = rng.nextInt( 50) + 10;
        Collection<Edge<Integer>> edges = new ArrayList<>();
        for (int e = 0; e < numConns; e++) {
            edges.add(new Edge<Integer>(limitSize, getConnection(limitSize, list.size(), rng), 1, true));
        }


        graph.addVertex(new Vertex<VocabWord>(limitSize, new VocabWord(1.0,String.valueOf(limitSize))), edges);



        log.info("Vertex 0 degree: {}", graph.getVertexDegree(0));
        log.info("Vertex 10k degree: {}", graph.getVertexDegree(limitSize));
        assertEquals(graph.getVertexDegree(0), graph.getVertexDegree(limitSize));
        assertTrue(graph.getConnectedVertexIndices(0).length > 4);
        assertArrayEquals(graph.getConnectedVertexIndices(0), graph.getConnectedVertexIndices(limitSize));

        return graph;
    }

    protected static int getConnection(int  currentIndex, int limit, Random rng) {
        return rng.nextInt(limit);
    }
}