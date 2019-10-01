import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
	    NeuralNet xor = new NeuralNet();
		xor.initializeTrainSet();

		//train the network for 100 trials
	    int epoch = 0;
	    for(int i = 0; i < 100; i++) {
			xor.initializeWeights();
	    	epoch += xor.train();
		}
	    epoch /= 100;
	    System.out.println("avg epoch " + epoch);
		xor.saveError();

		//save and load weights
		xor.save(new File("weights.txt"));
		try {
			xor.load("weights.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

