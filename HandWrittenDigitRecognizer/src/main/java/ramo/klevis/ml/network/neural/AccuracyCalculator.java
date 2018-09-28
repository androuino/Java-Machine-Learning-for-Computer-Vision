package ramo.klevis.ml.network.neural;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Klevis Ramo.
 */
//@Slf4j
public class AccuracyCalculator implements ScoreCalculator<MultiLayerNetwork> {

    private static final Logger log = LoggerFactory.getLogger(AccuracyCalculator.class);
    private final MnistDataSetIterator dataSetIterator;

    public AccuracyCalculator(MnistDataSetIterator dataSetIterator) {
        this.dataSetIterator = dataSetIterator;
    }

    private int i = 0;

    @Override
    public double calculateScore(MultiLayerNetwork network) {
        Evaluation evaluate = network.evaluate(dataSetIterator);
        double accuracy = evaluate.accuracy();
        log.info("Accuracy at iteration" + i++ + " " + accuracy);
        return 1 - evaluate.accuracy();
    }
}
