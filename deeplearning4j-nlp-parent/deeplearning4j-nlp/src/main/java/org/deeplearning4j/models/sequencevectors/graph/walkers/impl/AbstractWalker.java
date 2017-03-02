package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class AbstractWalker<T extends SequenceElement> implements GraphWalker<T> {
    @Getter
    protected IGraph<T, ? extends Number> sourceGraph;

    @Override
    public void buildVocabulary(VocabCache<T> vocabCache) {
        log.info("Building graph nodes vocabulary...");

        for (int n = 0; n < sourceGraph.numVertices(); n++) {
            T element = sourceGraph.getVertex(0).getValue();
            element.setElementFrequency(sourceGraph.getVertexDegree(n));

            vocabCache.addToken(element);
        }

        log.info("Building Huffman tree...");
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);
    }


}
