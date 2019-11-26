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
    
    /**
    * Trains the KNN classifier using an Inverted Index as
    * described in the class slides.
    *
    * @param trainExamples The vector of training examples
    */
    public void train(List<Example> trainExamples) {
        // Make an Inverted Index
        this.index = new InvertedIndex(trainExamples);
        // Keep track of what File corresponds to what category.
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
        // Get HM Vector.
        HashMapVector vector = testExample.getHashMapVector();
        // Retrieve all retrievals of test HM Vector.
        Retrieval[] retrievals = this.index.retrieve(vector);
        // Initialize results.
        double[] results = new double[this.categories.length];
        // Take top K retrievals.
        for (int i = 0; i < K; i++) {
            if (i < retrievals.length) {
                // Get file
                File file = retrievals[i].docRef.file;
                // Find Category
                Integer category = this.documentToCategory.get(file);
                if (category == null) {
                    System.out.println("Error finding category for " + retrievals[i].docRef);
                    continue;
                }
                // Increment count of category
                results[category] += 1.0;
            }
        }
        // Choose the category that appeared the most.
        // If tie, choose randomly between the tied categories.
        int categoryIndex = this.argMax(results);
        return categoryIndex == testExample.getCategory();
    }
}
