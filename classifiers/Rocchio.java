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
    public InvertedIndex index;

    protected boolean neg;


    /**
    * Create a new K Nearest Neighbors classifier with these attributes
    *
    * @param categories The array of Strings containing the category names
    * @param neg True if we are using modified version of Rocchio
    */
    public Rocchio(String[] categories, neg) {
        this.categories = categories;
        this.prototypes = new HashMapVector[categories.length];
        this.neg = neg;
        this.index = null;
    }

    /**
    * Create a new K Nearest Neighbors classifier with these attributes
    *
    * @param categories The array of Strings containing the category names
    * @param neg True if we are using modified version of Rocchio
    */
    public Rocchio(String[] categories) {
        this.categories = categories;
        this.prototypes = new HashMapVector[categories.length];
        this.neg = false;
        this.index = null;
    }

    /**
    * Returns the name
    */
    public String getName() {
        return this.name;
    }

    /**
    * Returns neg
    */
    public int getNeg() {
        return this.neg;
    }
    
    /**
    * Trains the Rocchio classifier - estimates the prior probs and calculates the
    * counts for each feature in different categories
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
            // Construct a new HM Vector to hold each terms TFIDF
            HashMapVector v = new HashMapVector();

            // For each Token in example HM Vector...
            for (Map.Entry<String, Weight> entry : exampleVector.entrySet()) {
                // Get token
                String token = entry.getKey();
                // Get TF of token
                int count = (int) entry.getValue().getValue();
                // Get TokenInfo of token, which holds idf of a token
                TokenInfo tokenInfo = this.index.tokenHash.get(token);
                // Compute TFIDF of token.
                v.increment(token, count * tokenInfo.idf);
            }
            // normalize
            double maxWeight = v.maxWeight();
            // Avoid divide by zero or infinity
            if (maxWeight == 0 || maxWeight == Double.NEGATIVE_INFINITY) {
                v.multiply(0);
            }
            else {
                v.multiply(1.0 / v.maxWeight());
            }
            // Cateogory index
            int categoryIndex = ex.getCategory();
            // Prototype at i is the prototype for Category i.
            if (this.neg) {
                // subtract for all other categories
                for (i = 0; i < this.prototypes.length; i++) {
                    if (i == categoryIndex) {
                        this.prototypes[i].add(v);
                    }
                    else {
                        this.prototypes[i].subtract(v);
                    }
                }
            }
            else {
                // default just add to category i
                this.prototypes[categoryIndex].add(v);
            }
        }
    }

    /**
    * Categorizes the test example using the trained KNN classifier, returning true if
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
            results[i] = vector.cosineTo(this.prototypes[i]);
        }
        // Choose Argmax
        int categoryIndex = this.argMax(results);

        // Take top K retrievals, and choose majority.
        return categoryIndex == testExample.getCategory();
    }
}