package ir.classifiers;

import java.util.*;

/**
 * Wrapper class to test KNN classifier using 10-fold CV.
 *
 * @author Rizwan Lubis
 */

public class TestKNN {
  /**
   * A driver method for testing the KNN classifier using
   * 10-fold cross validation.
   *
   * @param args a list of command-line arguments.  Specifying "-debug"
   *             will provide detailed output
   */
  public static void main(String args[]) throws Exception {
    String dirName = "/u/mooney/ir-code/corpora/curlie-science/";
    String[] categories = {"bio", "chem", "phys"};
    System.out.println("Loading Examples from " + dirName + "...");
    List<Example> examples = new DirectoryExamplesConstructor(dirName, categories).getExamples();
    System.out.println("Initializing K Nearest Neighbors classifier...");
    KNN BC;
    Integer K = 5;

    // setting debug flag gives very detailed output, suitable for debugging
    if (args.length == 2 && args[0].equals("-K"))
        K = Integer.parseInt(args[1]);
    BC = new KNN(categories, K);

    // Perform 10-fold cross validation to generate learning curve
    CVLearningCurve cvCurve = new CVLearningCurve(BC, examples);
    cvCurve.run();
  }
}
