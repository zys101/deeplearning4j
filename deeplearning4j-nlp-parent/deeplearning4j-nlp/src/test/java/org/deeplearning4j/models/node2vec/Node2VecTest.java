package org.deeplearning4j.models.node2vec;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
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
        Graph<VocabWord, Integer> graph = getGraph();
        GraphWalker<VocabWord> walker = new NearestVertexWalker.Builder<VocabWord>(graph)
                .setSamplingMode(SamplingMode.MAX_POPULARITY)
                .setDepth(2)
                .build();

        Node2Vec<VocabWord, Integer> node2Vec = new Node2Vec.Builder<VocabWord, Integer>(walker)
//                .elementsLearningAlgorithm(new CBOW<VocabWord>())
                .sequenceLearningAlgorithm(new DM<VocabWord>())
                .seed(119)
                .learningRate(0.025)
//                .iterations(100)
                .epochs(5)
                .trainElementsRepresentation(true)
                .build();

        node2Vec.fit();

        double simAZ = node2Vec.similarity("1","10000");
        double simAB = node2Vec.similarity("1","2");

        log.info("1 -> 10000 similarity: {}", simAZ);
        log.info("1 -> 2 similarity: {}", simAB);
    }

    /**
     * This method returns random generated
     *
     * @return
     */
    protected static Graph<VocabWord, Integer> getGraph() {
        List<Vertex<VocabWord>> list = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            list.add(new Vertex<VocabWord>(i, new VocabWord(1.0, String.valueOf(i))));
        }

        Graph<VocabWord, Integer> graph = new Graph<>(list);
        Random rng = new Random(seed);

        for (int i = 0; i < 10000; i++) {
            int conns = rng.nextInt( 10) + 5;

            for (int e = 0; e < conns; e++) {
                graph.addEdge(i, getConnection(i, list.size(), rng), 1, true);
            }
        }

        rng = new Random(seed);
        int numConns = rng.nextInt( 10) + 5;
        Collection<Edge<Integer>> edges = new ArrayList<>();
        for (int e = 0; e < numConns; e++) {
            edges.add(new Edge<Integer>(10000, getConnection(10000, list.size(), rng), 1, true));
        }


        graph.addVertex(new Vertex<VocabWord>(10000, new VocabWord(1.0,String.valueOf(10000))), edges);



        log.info("Vertex 0 degree: {}", graph.getVertexDegree(0));
        log.info("Vertex 10k degree: {}", graph.getVertexDegree(10000));
        assertEquals(graph.getVertexDegree(0), graph.getVertexDegree(10000));
        assertTrue(graph.getConnectedVertexIndices(0).length > 4);
        assertArrayEquals(graph.getConnectedVertexIndices(0), graph.getConnectedVertexIndices(10000));

        return graph;
    }

    protected static int getConnection(int  currentIndex, int limit, Random rng) {
        return rng.nextInt(limit);
    }
}