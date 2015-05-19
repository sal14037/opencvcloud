import hipi.image.FloatImage;

import org.apache.log4j.Logger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Utils {

	public static Logger l = Logger.getLogger(OCVCMapper.class);

	
	// convert floatImage to OpenCV Mat
	public static Mat floatImgToMat(FloatImage value) {
		float[] f = value.getData();
		int w = value.getWidth();
		int h = value.getHeight();
		int b = value.getBands();

		Mat m = new Mat(h, w, CvType.CV_32FC3);

		// Traverse image pixel data in raster-scan order and update running
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				float[] fv = { f[(j * w + i) * 3 + 2] * 255,
						f[(j * w + i) * 3 + 1] * 255,
						f[(j * w + i) * 3 + 0] * 255 };
				m.put(j, i, fv);
			}
		}

		return m;
	}

	// convert OpenCV Mat to byte array
	public static byte[] matToByteArr(Mat m) {
		MatOfByte buf = new MatOfByte();
		Imgcodecs.imencode(".jpg", m, buf);
		m.convertTo(m, CvType.CV_8UC1);
		byte[] b = buf.toArray();
		return b;
	}

	// crop the input matrix to specified rect
	public static Mat cropMatrix(Mat input, Rect rect) {
		Mat output = new Mat();
		// check if the position is within the image
		// area
		if (rect.x >= 0 && rect.y >= 0 && rect.x <= input.cols()
				&& rect.y <= input.rows()) {
			if (rect.x + rect.width <= input.cols()
					&& rect.y + rect.height <= input.rows()) {
				// cropping within boundaries
				l.info("Person cropping within limits");
				output = input.submat(rect.y, rect.y + rect.height, rect.x,
						rect.x + rect.width);
			} else {
				// cropping outside of boundaries
				l.info("Person cropping outside limits");
				output = input.submat(rect.x, input.rows(), rect.y,
						input.cols());
			}
		}
		return output;
	}
	
	// draw a rectangle
	public static void drawRect(Mat mat, Point p1, Point p2, Scalar colour) {
		Imgproc.rectangle(mat, p1, p2, colour, 2);
	}

	//draw the text
	public static void drawText(Mat mat, String text, Point p1, Scalar colour) {
		Imgproc.putText(mat, text, p1, Core.FONT_HERSHEY_PLAIN, 1, colour, 2,
				Core.LINE_AA, false);
	}
}
