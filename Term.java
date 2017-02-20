package org.deeplearning4j.examples.word2vec;

public class Term {
	public String term;
	public double score;
	
	public Term(String t, double s) {
		term = t;
		score = s;
	}
	
	public Term(String t) {
		term = t;
		score = Double.NEGATIVE_INFINITY;
	}
}