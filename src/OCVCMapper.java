import hipi.image.FloatImage;
import hipi.image.ImageHeader;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.Priority;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.Face;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

public class OCVCMapper extends
		Mapper<ImageHeader, FloatImage, Text, IntWritable> {

	public Path path;
	public FileSystem fileSystem;
	public Logger l = Logger.getLogger(this.getClass());

	public void setup(Context jc) throws IOException {
		Configuration conf = jc.getConfiguration();
		fileSystem = FileSystem.get(conf);
		path = new Path("output");
		fileSystem.mkdirs(path);
		l.setLevel(Level.INFO);
	}

	public void map(ImageHeader key, FloatImage value, Context context)
			throws IOException, InterruptedException {
		// measuring start time
		final long startTime = System.currentTimeMillis();

		// Convert float image input to OpenCV mat
		Mat cv = Utils.floatImgToMat(value);

		// Init People detection
		PeopleDetection p = new PeopleDetection();
		int framesWithPeople = 0;
		Point rectPoint1 = new Point();
		Point rectPoint2 = new Point();
		Point fontPoint = new Point();

		// Init Face detection
		FaceDetection fd = new FaceDetection();
		Mat mRgba = new Mat();
		Mat mGrey = new Mat();

		// Init Face recognition
		FaceRecognition fr = new FaceRecognition();

		if (!cv.empty()) {
			p.detectPeople(cv);
			if (p.getFoundLocations().rows() > 0) {
				int i = 0;

				// copy input_image to other matrices
				Mat toCrop = new Mat();
				cv.copyTo(toCrop);

				if (!p.allIsLost()) {
					for (Rect rect : p.getRectList()) {
						// saving the positions of the detected person in a
						// Point
						rectPoint1.x = rect.x;
						rectPoint1.y = rect.y;
						rectPoint2.x = rect.x + rect.width;
						rectPoint2.y = rect.y + rect.height;
						// location for text
						fontPoint.x = rect.x;
						fontPoint.y = rect.y - 4;
						// CHECKSTYLE:ON MagicNumber
						l.info("Weight of detection:"
								+ p.getWeightList().get(i).doubleValue());
						if (p.withinThreshold(i)) {
							framesWithPeople++;
							l.info("Face detection inside of HOG rectangle");

							// crop image area
							mRgba = Utils.cropMatrix(toCrop, rect);
							mGrey = fd.enhance(mRgba);

							// detect faces
							fd.detectFace(mGrey);
							if (!fd.faces.empty()) {
								for (Rect rect1 : fd.faces.toArray()) {
									// draw rectangle plus text
									if (fr.getLabel() > 0) {
										Imgproc.putText(cv,
												"Person " + fr.getLabel(),
												new Point(rect.x + rect1.x,
														rect.y + rect1.y - 5),
												Core.FONT_HERSHEY_PLAIN, 1, fr
														.getColour(), 2,
												Core.LINE_AA, false);
										Imgproc.rectangle(cv, new Point(rect.x
												+ rect1.x, rect.y + rect1.y),
												new Point(rect.x + rect1.x
														+ rect1.width, rect.y
														+ rect1.y
														+ rect1.height),
												fr.getColour(), 2);
									} else {
										Imgproc.putText(cv, "?", new Point(
												rect.x + rect1.x, rect.y
														+ rect1.y - 5),
												Core.FONT_HERSHEY_PLAIN, 1,
												new Scalar(0, 255, 255), 2,
												Core.LINE_AA, false);
										Imgproc.rectangle(cv, new Point(rect.x
												+ rect1.x, rect.y + rect1.y),
												new Point(rect.x + rect1.x
														+ rect1.width, rect.y
														+ rect1.y
														+ rect1.height),
												new Scalar(0, 255, 255), 2);
									}
								}
							}
							// Draw rectangle around found object
							Imgproc.rectangle(cv, rectPoint1, rectPoint2,
									p.rectColor, 2);
							Imgproc.putText(cv, String.format("%1.2f", p
									.getWeightList().get(i++)), fontPoint,
									Core.FONT_HERSHEY_PLAIN, 1.5, p.rectColor,
									2, Core.LINE_AA, false);
						}
						if (i < p.getWeightList().size() - 1) {
							i++;
						}
					}
				} else {
					// if no person is detected
					l.info("Face detection whole picture");
					mGrey = fd.enhance(cv);
					fd.detectFace(mGrey);
					if (fd.faces.rows() > 0) {
						for (Rect rect1 : fd.faces.toArray()) {
							Imgproc.rectangle(cv, new Point(rect1.x, rect1.y),
									new Point(rect1.x + rect1.width, rect1.y
											+ rect1.height), new Scalar(0, 255,
											255), 2);
							Imgproc.putText(cv, "?", new Point(rect1.x,
									rect1.y - 5), Core.FONT_HERSHEY_PLAIN, 1,
									new Scalar(0, 255, 255), 2, Core.LINE_AA,
									false);
						}
					}
				}
			} else {
				// if no person is detected
				l.info("Face detection whole picture");
				mGrey = fd.enhance(cv);
				fd.detectFace(mGrey);
				if (fd.faces.rows() > 0) {
					for (Rect rect1 : fd.faces.toArray()) {
						Imgproc.rectangle(cv, new Point(rect1.x, rect1.y),
								new Point(rect1.x + rect1.width, rect1.y
										+ rect1.height),
								new Scalar(0, 255, 255), 2);
						Imgproc.putText(cv, "?",
								new Point(rect1.x, rect1.y - 5),
								Core.FONT_HERSHEY_PLAIN, 1, new Scalar(0, 255,
										255), 2, Core.LINE_AA, false);
					}
				}
			}
			// Display the image
			Imgproc.putText(cv, key.getEXIFInformation().toString(), new Point(
					cv.width() - 380, cv.height() - 10),
					Core.FONT_HERSHEY_PLAIN, 1.5, p.fontColor, 2, Core.LINE_AA,
					false);
			l.info(System.currentTimeMillis() - startTime);
			Path outpath = new Path(path + "/" + value.hex() + ".jpg");
			FSDataOutputStream os = fileSystem.create(outpath);
			byte[] data = Utils.matToByteArr(cv);
			os.write(data);
			os.flush();
			os.close();
		}
		context.write(new Text(value.hex()), new IntWritable(framesWithPeople));
	}
}
