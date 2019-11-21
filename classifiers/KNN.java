package ir.classifiers;

import java.io.*;
import java.util.*;

import ir.vsr.*;
import ir.utilities.*;

/**
 * Class that extends Classifer to perform K Nearest Neighbors algorithm.
 *
 * @author Rizwan Lubis
 */

public class KNN extends Classifier {

    public static final String name = "KNN";
    protected int K = 5;
    public InvertedIndex index;

    protected HashMap<File, Integer> documentToCategory;


    /**
    * Create a new K Nearest Neighbors classifier with these attributes
    *
    * @param categories The array of Strings containing the category names
    */
    public KNN(String[] categories, int K) {
        this.categories = categories;
        this.K = K;
        this.index = null;
        this.documentToCategory = new HashMap<File, Integer>();
    }

    /**
    * Create a new K Nearest Neighbors classifier with these attributes
    *
    * @param categories The array of Strings containing the category names
    */
    public KNN(String[] categories) {
        this.categories = categories;
        this.index = null;
        this.documentToCategory = new HashMap<File, Integer>();
    }

    /**
    * Returns the name
    */
    public String getName() {
        return this.name;
    }

    public int getK() {
        return this.K;
    }
    
    /**
    * Trains the KNN classifier - estimates the prior probs and calculates the
    * counts for each feature in different categories
    *
    * @param trainExamples The vector of training examples
    */
    public void train(List<Example> trainExamples) {
        this.index = new InvertedIndex(trainExamples);
        for (Example example: trainExamples) {
            this.documentToCategory.put(example.getDocument().file, example.getCategory());
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
        Retrieval[] retrievals = this.index.retrieve(vector);
        double[] results = new double[this.categories.length];
        for (int i = 0; i < K; i++) {
            if (i < retrievals.length) {
                File file = retrievals[i].docRef.file;
                Integer category = this.documentToCategory.get(file);
                if (category == null) {
                    System.out.println("Error finding category for " + retrievals[i].docRef)
                    continue;
                }
                results[category] += 1.0;
            }
        }
        int categoryIndex = this.argMax(results);
        // Take top K retrievals, and choose majority.
        return categoryIndex == testExample.getCategory();
    }
}