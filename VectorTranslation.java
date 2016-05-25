// Author: Jose Manuel Gomez-Perez, Expert System

package org.deeplearning4j.examples.word2vec;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class VectorTranslation {
	public String dictionaryFile;
	public int dictionaryLength;
	public int columns;
	private static final String dicseparator=":";
	
	// Constructor
	public VectorTranslation(String dic, int length, int cols) {
		super();
		dictionaryFile = dic;
		dictionaryLength = length;
		columns = cols;
	}
		
	// Create a vector matrix from an array of words in the dictionary and the corresponding set of word2vec vectors	 
	public DoubleMatrix createVectorMatrix(String[] words, WordVectors v) {
		DoubleMatrix m = new DoubleMatrix(words.length, columns);
		for (int i = 0; i < words.length; i ++) {
			m.putRow(i, new DoubleMatrix(v.getWordVector(words[i])));
		}
		return m;
	}
		 
	// Create the translation matrix from the word2vec vectors of the source language, e.g. Spanish and the target language, e.g. English
	public DoubleMatrix calculateTranslationMatrix (WordVectors ves, WordVectors ven) throws IOException {
		FileReader reader = new FileReader(dictionaryFile);
		BufferedReader bufReader = new BufferedReader(reader);
		String line = bufReader.readLine();
		String[] pair = line.split(dicseparator);
		int count = 0;
		String[] source_training_set = new String[dictionaryLength];
		String[] target_training_set = new String[dictionaryLength];
		// Reading dictionary from a text file where each line has the format term_in_language_A:equivalent_term_in_language_B
		while (line != null && count < dictionaryLength) {
			String wes = pair[0];
			String wen = pair[1];
			// If word not in source or target vector, then do not include in the source and target training vectors
			if (ves.hasWord(wes) && ven.hasWord(wen)) {
				source_training_set[count] = wes;
				target_training_set[count] = wen;
				count++;	
			}
			line = bufReader.readLine();
			pair = line.split(dicseparator);
		}
		bufReader.close();           
		// Generate vector matrix for source and target training sets. For simplification, assuming dimension of target vectors is equal to the dimension of the source vectors. Some (minimal) changes may be required otherwise
		// WX=Z -> W=transpose(pinv(X)Z)
		DoubleMatrix matrix_train_source = createVectorMatrix(source_training_set, ves);
		DoubleMatrix matrix_train_target = createVectorMatrix(target_training_set, ven);
		DoubleMatrix pinverse_matrix = Solve.pinv(matrix_train_source);
		DoubleMatrix translationMatrix = pinverse_matrix.mmul(matrix_train_target).transpose();
		return translationMatrix;
	}

	public ArrayList<String> getNMostSimilarByVector(int n, String esw, WordVectors ven, double[] v) {
		//Target language vectors lookup
		ArrayList<String> candidates = new ArrayList<String>();			
		ArrayList<Term> arr = new ArrayList<Term>();
		int numbEnglishWords = ven.vocab().numWords();
		Double similarity = 0.0;
		for (int i=0; i < numbEnglishWords; i++) {
			String w = ven.vocab().wordAtIndex(i);
			double[] wordVector_en = ven.getWordVector(w);
			Double simAux = cosineSimilarity(v, wordVector_en);
			if (simAux > similarity) {
				similarity = simAux;
			}
			Term t = new Term(w,simAux);
			arr.add(t);
		}			 
		Collections.sort(arr, new Comparator<Term>() {
			@Override
			public int compare(Term t1, Term t2) {
				// Sort from max to min
				return new Double(t2.score).compareTo(new Double(t1.score));					 
			}
		});
		System.out.println("-----Closest Words to spanish word " + esw + " in English: ");
		for (int i=0; i <n && i < arr.size(); i++) {
			String term = arr.get(i).term;
			candidates.add(term);
			System.out.println(term);
		}
		System.out.println("--Score: " + similarity);
		return candidates;
	}
	
	// Naive but useful implementation of cosineSimilarity
	public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
		double dotProduct = 0.0;
		double normA = 0.0;
		double normB = 0.0;
		for (int i = 0; i < vectorA.length; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			normA += Math.pow(vectorA[i], 2);
			normB += Math.pow(vectorB[i], 2);
		}   
		return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}
}

