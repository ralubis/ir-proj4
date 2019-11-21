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
        this.index = new InvertedIndex(trainExamples);
        for (int i = 0; i < this.prototypes.length; i++) {
            this.prototypes[i] = new HashMapVector();
        }

        for (Example ex: trainExamples) {
            HashMapVector exampleVector = ex.getHashMapVector();
            HashMapVector v = new HashMapVector();
            for (Map.Entry<String, Weight> entry : exampleVector.entrySet()) {
                String token = entry.getKey();
                int count = (int) entry.getValue().getValue();
                TokenInfo tokenInfo = this.index.tokenHash.get(token);
                v.increment(token, count * tokenInfo.idf);
            }
            // normalize
            double maxWeight = v.maxWeight();
            if (maxWeight == 0 || maxWeight == Double.NEGATIVE_INFINITY) {
                v.multiply(0);
            }
            else {
                v.multiply(1.0 / v.maxWeight());
            }
            int categoryIndex = ex.getCategory();
            if (this.neg) {
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
                this.prototypes[i].add(v);
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
        HashMapVector vector = testExample.getHashMapVector();
        double[] results = new double[this.prototypes.length];
        for (int i = 0; i < this.prototypes.length; i++) {
            results[i] = vector.cosineTo(this.prototypes[i]);
        }
        int categoryIndex = this.argMax(results);

        // Take top K retrievals, and choose majority.
        return categoryIndex == testExample.getCategory();
    }
}