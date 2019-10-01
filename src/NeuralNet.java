import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {
    //hyper-parameters
    private boolean binary = true;
    private int numInputs = 2;
    private int numHidden = 4;
    private int numOutputs = 1;
    private double learningRate = 0.2;
    private double momentum = 0.9;
    private double a = 0;
    private double b = 1;
    private double initCeiling = 0.5;
    private double initFloor = -0.5;
    private static double errorThreshold = 0.05;

    //layers
    private double[] inputLayer = new double[numInputs + 1];  //one extra bias node
    private double[] hiddenLayer = new double[numHidden + 1];
    private double[] outputLayer = new double[numOutputs];


    //weights
    private double[][] w1 = new double[numInputs + 1][numHidden];
    private double[][] w2 = new double[numHidden + 1][numOutputs];

    //back-propagation update arrays
    private double[] deltaOutput = new double[numOutputs];
    private double[] deltaHidden = new double[numHidden];

    private double[][] deltaW1 = new double[numInputs + 1][numHidden];
    private double[][] deltaW2 = new double[numHidden + 1][numOutputs];

    //error data
    private double[] totalError = new double[numOutputs];
    private double[] singleError = new double[numOutputs];

    private List<String> errorList = new LinkedList<>();

    //training set
    private double[][] trainX; //one bias node
    private double[][] trainY;

    public NeuralNet(int numInputs, int numHidden, int numOutputs, double learningRate, double momentum, double a, double b) {
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.a = a;
        this.b = b;
    }

    public NeuralNet() {
    }

    public void initializeTrainSet() {
        if (binary) {
            trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            trainY = new double[][]{{0}, {1}, {1}, {0}};
        } else {
            trainX = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
            trainY = new double[][]{{-1}, {1}, {1}, {-1}};
        }
    }

    //bipolar sigmoid
    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    @Override
    public double customSigmoid(double x) {
        if (!binary) {
            b = 1;
            a = -1;
        }
        return (b - a) / (1 + Math.exp(-x)) + a;
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHidden; j++) { //no weights for the bias node
                double r = new Random().nextDouble();
                w1[i][j] = initFloor + (r * (initCeiling - initFloor));
                deltaW1[i][j] = 0.0;
            }
        }

        for (int j = 0; j < numHidden + 1; j++) {
            for (int k = 0; k < numOutputs; k++) {
                double r = new Random().nextDouble();
                w2[j][k] = initFloor + (r * (initCeiling - initFloor));
                deltaW2[j][k] = 0.0;
            }
        }
    }

    @Override
    public void zeroWeights() {
        //w1, w2 entries are automatically assigned default zero by compiler
    }

    private void initializeLayers(double[] sample) {
        for (int i = 0; i < numInputs; i++) {
            inputLayer[i] = sample[i];
        }
        inputLayer[numInputs] = 1;
        hiddenLayer[numHidden] = 1;
    }

    private void forwardPropagation(double[] sample) {
        initializeLayers(sample);
        for (int j = 0; j < numHidden; j++) {
            hiddenLayer[j] = 0;
            for (int i = 0; i < numInputs + 1; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
            hiddenLayer[j] = customSigmoid(hiddenLayer[j]);
        }

        for (int k = 0; k < numOutputs; k++) {
            outputLayer[k] = 0;
            for (int j = 0; j < numHidden + 1; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            outputLayer[k] = customSigmoid(outputLayer[k]);

        }
        //System.out.println(singleError[0]);
    }

    private void backPropagation() {
        //compute deltaOutput[]
        for (int k = 0; k < numOutputs; k++) {
            deltaOutput[k] = 0;
            deltaOutput[k] = binary ? singleError[k] * outputLayer[k] * (1 - outputLayer[k]) :
                    singleError[k] * (outputLayer[k] + 1) * 0.5 * (1 - outputLayer[k]);
        }

        //update w2
        for (int k = 0; k < numOutputs; k++) {
            for (int j = 0; j < numHidden + 1; j++) {
                deltaW2[j][k] = momentum * deltaW2[j][k] + learningRate * deltaOutput[k] * hiddenLayer[j];
                w2[j][k] += deltaW2[j][k];
            }
        }

        //Compute deltaHidden
        for (int j = 0; j < numHidden; j++) {
            deltaHidden[j] = 0;
            for (int k = 0; k < numOutputs; k++) {
                deltaHidden[j] += w2[j][k] * deltaOutput[k];
            }
            deltaHidden[j] = binary ? deltaHidden[j] * hiddenLayer[j] * (1 - hiddenLayer[j]) :
                    deltaHidden[j] * (hiddenLayer[j] + 1) * 0.5 * (1 - hiddenLayer[j]);
        }


        //Update w1
        for (int j = 0; j < numHidden; j++) {
            for (int i = 0; i < numInputs + 1; i++) {
                deltaW1[i][j] = momentum * deltaW1[i][j]
                        + learningRate * deltaHidden[j] * inputLayer[i];
                w1[i][j] += deltaW1[i][j];
            }
        }

    }

    public int train() {
        errorList.clear();
        int epoch = 0;

        do {
            for (int k = 0; k < numOutputs; k++) {
                totalError[k] = 0;
            }
            int numSamples = trainX.length;
            for (int i = 0; i < numSamples; i++) {
                double[] sample = trainX[i];
                forwardPropagation(sample);
                for (int k = 0; k < numOutputs; k++) {
                    singleError[k] = trainY[i][k] - outputLayer[k];
                    totalError[k] += Math.pow(singleError[k], 2);
                }
                backPropagation();
            }

            for (int k = 0; k < numOutputs; k++) {
                totalError[k] /= 2;
                System.out.println("Total error for output number " + (k + 1) + ": " + totalError[k]);
            }
            errorList.add(Double.toString(totalError[0]));
            epoch++;
        } while (totalError[0] > errorThreshold);
        System.out.println("This trial epoch " + epoch + "\n");
        return epoch;
    }

    @Override
    public double outputFor(double[] X) {
        forwardPropagation(X);
        return outputLayer[0];
    }

    @Override
    public double train(double[] X, double argValue) {
        forwardPropagation(X);
        return argValue - outputLayer[0];
    }

    @Override
    public void save(File argFile) {
        try {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < w1.length; i++) {
                for (int j = 0; j < w1[0].length; j++) {
                    builder.append(w1[i][j] + " ");
                }
                builder.append("\n");
            }
            builder.append("\n");
            for (int i = 0; i < w2.length; i++) {
                for (int j = 0; j < w2[0].length; j++) {
                    builder.append(w2[i][j] + " ");
                }
                builder.append("\n");
            }
            Files.write(argFile.toPath(), builder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void load(String argFileName) throws IOException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader("./weights.txt")));
        double[][] w1 = new double[numInputs + 1][numHidden];
        double[][] w2 = new double[numHidden + 1][numOutputs];
        boolean readingW1 = true;
        int lineIndex = 0;
        while (sc.hasNextLine()) {
            if (readingW1) {
                String[] line = sc.nextLine().trim().split(" ");
                if (line[0].length() == 0) {
                    readingW1 = false;
                    lineIndex = 0;
                    continue;
                }
                //System.out.println(line[0]);
                for (int j = 0; j < line.length; j++) {
                    w1[lineIndex][j] = Double.parseDouble(line[j]);
                }
                lineIndex++;
            } else {
                String[] line = sc.nextLine().trim().split(" ");
                if (line[0].length() == 0) {
                    break;
                }
                //System.out.println(line[0]);
                for (int j = 0; j < line.length; j++) {
                    w2[lineIndex][j] = Double.parseDouble(line[j]);
                }
                lineIndex++;
            }
        }
    }

    public void saveError() {
        try {
            Files.write(Paths.get("./trainError.txt"), errorList);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
