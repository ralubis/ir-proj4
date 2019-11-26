package ir.classifiers;

import java.io.*;
import java.util.*;

import ir.vsr.*;
import ir.utilities.*;

/**
 * Class that extends Classifer using the Rocchio algorithm.
 *
 * @author Rizwan Lubis
 */

public class Rocchio extends Classifier {

    public static final String name = "Rocchio";
    protected HashMapVector[] prototypes;
    protected double[] prototypeLengths;
    public InvertedIndex index;

    protected boolean neg;


    /**
    * Create a new Rocchio classifier with these attributes
    *
    * @param categories The array of Strings containing the category names
    * @param neg True if we are using modified version of Rocchio
    */
    public Rocchio(String[] categories, boolean neg) {
        this.categories = categories;
        this.prototypes = new HashMapVector[categories.length];
        this.prototypeLengths = new double[categories.length];
        this.neg = neg;
        this.index = null;
    }

    /**
    * Returns the name
    */
    public String getName() {
        return this.name;
    }
    
    /**
    * Trains the Rocchio classifier - creates the HashMapVectors
    * of each category prototype.
    *
    * @param trainExamples The vector of training examples
    */
    public void train(List<Example> trainExamples) {
        // index to find IDFs
        this.index = new InvertedIndex(trainExamples);
        
        // Initialize prototypes
        for (int i = 0; i < this.prototypes.length; i++) {
            this.prototypes[i] = new HashMapVector();
        }

        // For each training example...
        for (Example ex: trainExamples) {
            // HM Vector holding only TF
            HashMapVector exampleVector = ex.getHashMapVector();
            // normalize
            double maxWeight = exampleVector.maxWeight();
            // Cateogory index
            int categoryIndex = ex.getCategory();
            // Avoid divide by zero or infinity
            if (maxWeight != 0 && maxWeight != Double.NEGATIVE_INFINITY) {
                // For each Token in example HM Vector...
                for (Map.Entry<String, Weight> entry : exampleVector.entrySet()) {
                    // Get token
                    String token = entry.getKey();
                    // Get TF of token
                    int count = (int) entry.getValue().getValue();
                    // Get TokenInfo of token, which holds idf of a token
                    TokenInfo tokenInfo = this.index.tokenHash.get(token);
                    // Compute TFIDF of token and normalize
                    double tfidf = tokenInfo.idf * count / maxWeight;
                    
                    // Update Prototypes
                    if (this.neg) {
                        for (int i = 0; i < this.prototypes.length; i++) {
                            if (i == categoryIndex) {
                                this.prototypes[i].increment(token, tfidf);
                            }
                            else {
                                this.prototypes[i].increment(token, -tfidf);
                            }
                        }
                    }
                    else {
                        this.prototypes[categoryIndex].increment(token, tfidf);
                    }
                }
            }
        }
        // Find lengths of prototypes.
        for (int i = 0; i < this.prototypes.length; i++) {
            this.prototypeLengths[i] = this.prototypes[i].length();
        }
    }

    /**
    * Categorizes the test example using the trained Rocchio classifier, returning true if
    * the predicted category is same as the actual category
    *
    * @param testExample The test example to be categorized
    */
    public boolean test(Example testExample) {
        // Get HM Vector
        HashMapVector vector = testExample.getHashMapVector();
        // Initialize Results
        double[] results = new double[this.prototypes.length];
        // Find Cosine Similarity to each prototype
        for (int i = 0; i < this.prototypes.length; i++) {
            results[i] = vector.cosineTo(this.prototypes[i], this.prototypeLengths[i]);
        }
        // Choose Argmax
        int categoryIndex = this.argMax(results);

        // Take top K retrievals, and choose majority.
        return categoryIndex == testExample.getCategory();
    }
}
