import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.examples.word2vec.VectorTranslation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class VectorTranslationTest {
		// This test tranaslates from vectors in Spanish to English, both extracted from the EU parliament corpus
		// The dictionary is produced by applying machine translation on the most frequent Spanish terms in the corpus and generating the English equivalent
		static final String sourceVector = "europarl-v7.es-en.es.vector";
		static final String targetVector = "europarl-v7.es-en.en.vector";
		static final String dictionaryFile = "europarl-v7.es-en-lemmas.dic";
		static final int dictionaryLength = 5000;
		static final int columns = 400;
		static final int n = 10;

		@Test
		public void testGetMatrix() throws IOException {
			// Reload target and source vectors
			WordVectors ves = WordVectorSerializer.loadTxtVectors(new File(sourceVector));	 
			//Load source and target training set from dictionary
			System.out.println("Source vector loaded");
			WordVectors ven = WordVectorSerializer.loadTxtVectors(new File(targetVector));
			System.out.println("Target vector loaded");
			VectorTranslation mapper = new VectorTranslation(dictionaryFile, dictionaryLength, columns);
			DoubleMatrix translationMatrix = mapper.calculateTranslationMatrix(ves, ven);
			//Example Spanish -> English
			String[] terms1 = {
					"ser",
					"haber",
					"espacio",
					"mostrar",
					"asesino",
					"intimidad",
					// Hey, I know the numbers, too!
					"dos", "tres", "cuatro", "sesenta",
					"honradez",
					"banquero",
					"medios",
					"deporte",
					"decidido"
			};
			for (String term : terms1) {
				DoubleMatrix vsource = new DoubleMatrix(ves.getWordVector(term));
		        double [] vtargetestimated = translationMatrix.mmul(vsource).transpose().toArray();
		        mapper.getNMostSimilarByVector(n, term, ven, vtargetestimated);
			}
		}
}

