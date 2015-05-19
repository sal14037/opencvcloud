import hipi.image.FloatImage;
import hipi.image.ImageHeader;

import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Random;

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
		Mapper<ImageHeader, FloatImage, IntWritable, Text> {

	public Path path;
	public FileSystem fileSystem;
	public Logger l = Logger.getLogger(this.getClass());

	public void setup(Context jc) throws IOException {
		// set outputpath
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
		Point faceRectPoint1 = new Point();
		Point faceRectPoint2 = new Point();

		// Init Face recognition
		FaceRecognition fr = new FaceRecognition();
		Mat face = new Mat();
		Point faceFontPoint = new Point();

		// Init Header extraction
		String text = "Person ";
		key.addEXIFInformation("Date", new Date().toLocaleString());
		Random r = new Random();
		key.addEXIFInformation("Capture device", "Camera" + r.nextInt(10));
		String date = key.getEXIFInformation("Date");
		String camera = key.getEXIFInformation("Capture device");

		// check if received frame is not empty
		if (!cv.empty()) {
			// detect People
			p.detectPeople(cv);
			// check if the amount of detected people is bigger than 0
			if (p.getFoundLocations().rows() > 0) {
				// init counter for weights
				int i = 0;

				// copy input_image to other matrices in order to keep original
				// frame
				Mat toCrop = new Mat();
				cv.copyTo(toCrop);

				// check if all the people detections are < threshold
				if (!p.allIsLost()) {
					// loop through people
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
						// check if weight of current person detection is within
						// the set threshold
						if (p.withinThreshold(i)) {
							framesWithPeople++;
							l.info("Face detection inside of HOG rectangle");

							// crop image area
							mRgba = Utils.cropMatrix(toCrop, rect);
							// enhance image
							mGrey = fd.enhance(mRgba);

							// detect faces
							fd.detectFace(mGrey);
							// check if any faces are detected
							if (!fd.faces.empty()) {
								// loop trough detected faces
								for (Rect rect1 : fd.faces.toArray()) {
									// crop image area to face area
									face = Utils.cropMatrix(mGrey, rect1);
									// start recognition process
									fr.loadRecogniser();
									fr.recognise(face);
									text = text + " " + fr.getLabel();
									// saving the positions of the detected
									// person in a Point
									faceRectPoint1.x = rect1.x;
									faceRectPoint1.y = rect1.y;
									faceRectPoint2.x = rect1.x + rect1.width;
									faceRectPoint2.y = rect1.y + rect1.height;
									// location for text
									faceFontPoint.x = rect1.x;
									faceFontPoint.y = rect1.y - 4;
									// check if recognition is valid
									if (fr.getLabel() > 0) {
										// draw rectangle plus text
										Utils.drawText(cv,
												"Person" + fr.getLabel(),
												faceFontPoint, fr.getColour());
										Utils.drawRect(cv, faceRectPoint1,
												faceRectPoint2, fr.getColour());
									} else {
										// draw rectangle plus text for unsure
										// detection
										Utils.drawText(cv, "?", faceFontPoint,
												new Scalar(0, 255, 255));
										Utils.drawRect(cv, faceRectPoint1,
												faceRectPoint2, new Scalar(0,
														255, 255));
									}
								}
							}
							// Draw rectangle around found object
							Utils.drawRect(cv, rectPoint1, rectPoint2,
									p.rectColor);
							Utils.drawText(cv, String.format("%1.2f", p
									.getWeightList().get(i)), fontPoint,
									p.rectColor);
						}
						if (i < p.getWeightList().size() - 1) {
							i++;
						}
					}
					// if no person is detected use whole image
				} else {
					l.info("Face detection whole picture, AllIsLost");
					// enhance whole image
					mGrey = fd.enhance(cv);
					// detect faces
					fd.detectFace(mGrey);
					if (fd.faces.rows() > 0) {
						for (Rect rect1 : fd.faces.toArray()) {
							face = Utils.cropMatrix(mGrey, rect1);
							fr.loadRecogniser();
							fr.recognise(face);
							// saving the positions of the detected person in a
							// Point
							faceRectPoint1.x = rect1.x;
							faceRectPoint1.y = rect1.y;
							faceRectPoint2.x = rect1.x + rect1.width;
							faceRectPoint2.y = rect1.y + rect1.height;
							// location for text
							faceFontPoint.x = rect1.x;
							faceFontPoint.y = rect1.y - 4;
							// check if recognition is valid
							if (fr.getLabel() > 0) {
								// draw rectangle plus text
								Utils.drawText(cv, "Person" + fr.getLabel(),
										faceFontPoint, fr.getColour());
								Utils.drawRect(cv, faceRectPoint1,
										faceRectPoint2, fr.getColour());
							} else {
								// draw rectangle plus text for unsure detection
								Utils.drawText(cv, "?", faceFontPoint,
										new Scalar(0, 255, 255));
								Utils.drawRect(cv, faceRectPoint1,
										faceRectPoint2, new Scalar(0, 255, 255));
							}
						}
					}
				}
				// if no person is detected use whole image
			} else {
				// if no person is detected
				l.info("Face detection whole picture, no people");
				// enhance whole image
				mGrey = fd.enhance(cv);
				// detect faces
				fd.detectFace(mGrey);
				if (fd.faces.rows() > 0) {
					for (Rect rect1 : fd.faces.toArray()) {
						face = Utils.cropMatrix(mGrey, rect1);
						fr.loadRecogniser();
						fr.recognise(face);
						faceRectPoint1.x = rect1.x;
						faceRectPoint1.y = rect1.y;
						faceRectPoint2.x = rect1.x + rect1.width;
						faceRectPoint2.y = rect1.y + rect1.height;
						// location for text
						faceFontPoint.x = rect1.x;
						faceFontPoint.y = rect1.y - 4;
						// check if recognition is valid
						if (fr.getLabel() > 0) {
							// draw rectangle plus text
							Utils.drawText(cv, "Person" + fr.getLabel(),
									faceFontPoint, fr.getColour());
							Utils.drawRect(cv, faceRectPoint1, faceRectPoint2,
									fr.getColour());
						} else {
							// draw rectangle plus text for unsure detection
							Utils.drawText(cv, "?", faceFontPoint, new Scalar(
									0, 255, 255));
							Utils.drawRect(cv, faceRectPoint1, faceRectPoint2,
									new Scalar(0, 255, 255));
						}
					}
				}
			}
			// Add header information to image
			Imgproc.putText(cv, camera + ", " + date, new Point(
					cv.width() - 380, cv.height() - 10),
					Core.FONT_HERSHEY_PLAIN, 1.5, p.fontColor, 2, Core.LINE_AA,
					false);
			// log execution time
			l.info(System.currentTimeMillis() - startTime);
			Path outpath = new Path(path + "/" + value.hex() + ".jpg");
			FSDataOutputStream os = fileSystem.create(outpath);
			//convert OpenCV Mat to byte data in order to store it
			byte[] data = Utils.matToByteArr(cv);
			os.write(data);
			os.flush();
			os.close();
		}
		// emit record
		context.write(new IntWritable(1), new Text(value.hex() + " " + camera
				+ " " + date + " " + text));
	}
}
