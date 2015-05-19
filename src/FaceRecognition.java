import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.face.Face;
import org.opencv.face.LBPHFaceRecognizer;

public class FaceRecognition {

	private LBPHFaceRecognizer bf = Face.createLBPHFaceRecognizer();
	//predicted result
	private int[] predicted = new int[1];
	// confidence of detection
	private double[] confidence = new double[1];
	private final Scalar colour = new Scalar(255, 0, 0);
	
	//specify stored FR location
	private String savedFR = "/home/thomas/workspace/opencvcloud/test.yml";

	// FACERECOGNITION

	//load recogniser from saved recogniser
	public void loadRecogniser() {
		bf.load(savedFR);
	}

	// recognition on input matrix
	public int recognise(Mat m) {
		bf.predict(m, predicted, confidence);
		return 0;
	}

	public int getLabel() {
		return predicted[0];
	}

	public double[] getConfidence() {
		return confidence;
	}

	public Scalar getColour() {
		return colour;
	}

}
