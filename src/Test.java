import hipi.image.FloatImage;
import hipi.image.ImageHeader;
import hipi.imagebundle.mapreduce.ImageBundleInputFormat;
import hipi.util.ByteUtils;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.util.List;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;
//import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
//import org.bytedeco.javacpp.opencv_core.MatVector;
//import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
//import static org.bytedeco.javacpp.opencv_contrib.*;
//import static org.bytedeco.javacpp.opencv_core.*;
//import static org.bytedeco.javacpp.opencv_highgui.*;

public class Test extends Configured implements Tool {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static class TestMapper extends
			Mapper<ImageHeader, FloatImage, Text, IntWritable> {

		public Path path;
		public FileSystem fileSystem;

		public void setup(Context jc) throws IOException {
			Configuration conf = jc.getConfiguration();
			fileSystem = FileSystem.get(conf);
			path = new Path("output");
			fileSystem.mkdirs(path);
		}

		public void map(ImageHeader key, FloatImage value, Context context)
				throws IOException, InterruptedException {
			float[] f = value.getData();

			Mat cv = new Mat(value.getHeight(), value.getWidth(),
					CvType.CV_32FC3);
			int w = value.getWidth();
			int h = value.getHeight();
			int b = value.getBands();

			// Traverse image pixel data in raster-scan order and update running
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					float[] fv = { f[(j * w + i) * 3 + 2] * 255,
							f[(j * w + i) * 3 + 1] * 255,
							f[(j * w + i) * 3 + 0] * 255 };
					cv.put(j, i, fv);
				}
			}
			cv.convertTo(cv, CvType.CV_8UC1);
			// People detection
			final HOGDescriptor hog = new HOGDescriptor();
			// final HOGDescriptor hog = new HOGDescriptor(new Size(128, 64),
			// new Size(16, 16), new Size(8, 8), new Size(8, 8), 9, 0, -1, 0,
			// 0.2, false, 64);
			// necessary matrices for descriptor
			final MatOfFloat descriptors = HOGDescriptor
					.getDefaultPeopleDetector();
			hog.setSVMDetector(descriptors);
			final MatOfRect foundLocations = new MatOfRect();
			final MatOfDouble foundWeights = new MatOfDouble();
			final Size winStride = new Size(8, 8);
			final Size padding = new Size(32, 32);
			final Point rectPoint1 = new Point();
			final Point rectPoint2 = new Point();
			final Point fontPoint = new Point();
			int frames = 0;
			int framesWithPeople = 0;
			final long startTime = System.currentTimeMillis();
			final Scalar rectColor = new Scalar(0, 255, 0);
			final Scalar fontColor = new Scalar(255, 255, 255);

			// Facedetection
			CascadeClassifier face_cascade = new CascadeClassifier(
					"input/haarcascade_frontalface_alt.xml");
			// // Facerecognition
			// String trainingDir =
			// "D:/workspace/JavaCVFaceRecognition/src/training/";
			// File root = new File(trainingDir);
			// FilenameFilter imgFilter = new FilenameFilter() {
			// public boolean accept(File dir, String name) {
			// name = name.toLowerCase();
			// return name.endsWith(".jpg") || name.endsWith(".pgm")
			// || name.endsWith(".png");
			// }
			// };
			// File[] imageFiles = root.listFiles(imgFilter);
			// org.bytedeco.javacpp.opencv_core.MatVector images = new
			// MatVector(
			// imageFiles.length);
			// org.bytedeco.javacpp.opencv_core.Mat labels = new
			// org.bytedeco.javacpp.opencv_core.Mat(
			// imageFiles.length, 1, CV_32SC1);
			// IntBuffer labelsBuf = labels.getIntBuffer();
			// int counter = 0;
			// for (File image : imageFiles) {
			// org.bytedeco.javacpp.opencv_core.Mat img = imread(
			// image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
			// int label = Integer.parseInt(image.getName().split("\\-")[0]);
			// images.put(counter, img);
			// labelsBuf.put(counter, label);
			// counter++;
			// }
			// FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
			// faceRecognizer.train(images, labels);

			if (!cv.empty()) {
				// hog.detectMultiScale(cv, foundLocations, foundWeights, 0.0,
				// winStride, padding, 1.05, 2.0, false);
				hog.detectMultiScale(cv, foundLocations, foundWeights);
				if (foundLocations.rows() > 0) {
					List<Double> weightList = foundWeights.toList();
					List<Rect> rectList = foundLocations.toList();
					int i = 0;
					// copy input_image to other matrices
					Mat mRgba = new Mat();
					Mat mGrey = new Mat();
					MatOfRect faces = new MatOfRect();
					Mat toCrop = new Mat();
					cv.copyTo(toCrop);

					boolean allwrong = true;
					for (Double d : weightList) {
						if (d >= 1) {
							allwrong = false;
							break;
						}
					}
					System.out.println("all is wrong:" + allwrong);

					if (!allwrong) {
						for (Rect rect : rectList) {
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
							System.out.println(weightList.get(i).doubleValue());
							if (weightList.get(i).doubleValue() >= 1) {
								framesWithPeople++;
								// Apply the classifier to the cropped image
								System.out.println("FD:HOG rectangle");
								System.out.println(weightList.get(i)
										.doubleValue());
								toCrop.copyTo(mRgba);
								toCrop.copyTo(mGrey);
								// check if the position is within the image
								// area
								if (rect.x >= 0 && rect.y >= 0
										&& rect.x <= cv.cols()
										&& rect.y <= cv.rows()) {
									if (rect.x + rect.width <= cv.cols()
											&& rect.y + rect.height <= cv
													.rows()) {
										// cropping within boundaries
										System.out
												.println("CROPPING within limits");
										mRgba = mRgba.submat(rect.y, rect.y
												+ rect.height, rect.x, rect.x
												+ rect.width);
										mGrey = mGrey.submat(rect.y, rect.y
												+ rect.height, rect.x, rect.x
												+ rect.width);
									} else {
										// cropping outside of boundaries
										System.out
												.println("CROPPING outside limit");
										mRgba = mRgba.submat(rect.x, cv.rows(),
												rect.y, cv.cols());
										mGrey = mGrey.submat(rect.x, cv.rows(),
												rect.y, cv.cols());
									}
								}
								// enhance image by combining grey and colour
								Imgproc.cvtColor(mRgba, mGrey,
										Imgproc.COLOR_BGR2GRAY);
								Imgproc.equalizeHist(mGrey, mGrey);
								// detect faces
								face_cascade.detectMultiScale(mGrey, faces);
								if (!faces.empty()) {
									for (Rect rect1 : faces.toArray()) {
										Imgproc.rectangle(cv, new Point(rect.x
												+ rect1.x, rect.y + rect1.y),
												new Point(rect.x + rect1.x
														+ rect1.width, rect.y
														+ rect1.y
														+ rect1.height),
												new Scalar(0, 255, 255), 2);
										// TODO JAVACV OPENCV MAT
										// FACERECOGNITION
										// byte[] bytearr =
										// facePanel.toByteArray(mRgba);
										// org.bytedeco.javacpp.opencv_core.Mat
										// facerec
										// = new
										// org.bytedeco.javacpp.opencv_core.Mat(
										// bytearr);
										// imwrite("output.png", facerec);
										// int predictedLabel = faceRecognizer
										// .predict(facerec);
										// System.out.println("Predicted label: "
										// + predictedLabel);
										// if (predictedLabel <= 1)
										Imgproc.putText(cv, "?", new Point(
												rect.x + rect1.x, rect.y
														+ rect1.y - 5),
												Core.FONT_HERSHEY_PLAIN, 1,
												new Scalar(0, 255, 255), 2,
												Core.LINE_AA, false);
									}
								}
								// Draw rectangle around found object
								Imgproc.rectangle(cv, rectPoint1, rectPoint2,
										rectColor, 2);
								Imgproc.putText(
										cv,
										String.format("%1.2f",
												weightList.get(i++)),
										fontPoint, Core.FONT_HERSHEY_PLAIN,
										1.5, fontColor, 2, Core.LINE_AA, false);
							}
							if (i < weightList.size() - 1) {
								i++;
							}
						}
					} else {
						// if no person is detected
						System.out.println("FD:whole picture");
						cv.copyTo(mRgba);
						cv.copyTo(mGrey);
						Imgproc.cvtColor(mRgba, mGrey, Imgproc.COLOR_BGR2GRAY);
						Imgproc.equalizeHist(mGrey, mGrey);
						MatOfInt reject = new MatOfInt();
						MatOfDouble weights = new MatOfDouble();
						face_cascade.detectMultiScale3(mGrey, faces, reject,
								weights);
						// TODO weighted list checkList
						if (faces.rows() > 0) {
							for (Rect rect1 : faces.toArray()) {
								Imgproc.rectangle(cv, new Point(rect1.x,
										rect1.y), new Point(rect1.x
										+ rect1.width, rect1.y + rect1.height),
										new Scalar(0, 255, 255), 2);
								Imgproc.putText(cv, "?", new Point(rect1.x,
										rect1.y - 5), Core.FONT_HERSHEY_PLAIN,
										1, new Scalar(0, 255, 255), 2,
										Core.LINE_AA, false);
							}
						}
					}
				} else {
					// if no person is detected
					System.out.println("FD:whole picture");
					Mat mRgba = new Mat();
					Mat mGrey = new Mat();
					MatOfRect faces = new MatOfRect();
					cv.copyTo(mRgba);
					cv.copyTo(mGrey);
					Imgproc.cvtColor(mRgba, mGrey, Imgproc.COLOR_BGR2GRAY);
					Imgproc.equalizeHist(mGrey, mGrey);
					MatOfInt reject = new MatOfInt();
					MatOfDouble weights = new MatOfDouble();
					face_cascade.detectMultiScale3(mGrey, faces, reject,
							weights);
					// TODO weighted list checkList
					if (faces.rows() > 0) {
						for (Rect rect1 : faces.toArray()) {
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
				// Display the image
				Imgproc.putText(cv, key.getEXIFInformation().toString(),
						new Point(cv.width() - 380, cv.height() - 10),
						Core.FONT_HERSHEY_PLAIN, 1.5, fontColor, 2,
						Core.LINE_AA, false);
				System.out.println(System.currentTimeMillis() - startTime);
				Path outpath = new Path(path + "/" + value.hex() + ".jpg");
				FSDataOutputStream os = fileSystem.create(outpath);

				MatOfByte buf = new MatOfByte();
				Imgcodecs.imencode(".jpg", cv, buf);
				cv.convertTo(cv, CvType.CV_8UC1);
				byte[] data = buf.toArray();
				os.write(data);
				os.flush();
				os.close();
			}
			context.write(new Text(value.hex()), new IntWritable(
					framesWithPeople));
		}
	}

	public static class TestReducer extends
			Reducer<Text, IntWritable, IntWritable, Text> {

		public void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			int total = 0;
			String result = " Detected faces: ";
			for (Text val : values) {
				// Emit output of job which will be written to HDFS
				context.write(key, new Text(key + result + val));
				total++;
			}
		}
	}

	public int run(String[] args) throws Exception {
		// Check input arguments
		if (args.length != 2) {
			System.out
					.println("Usage: firstprog <input HIB> <output directory>");
			System.exit(0);
		}

		// Initialize and configure MapReduce job
		Job job = Job.getInstance();
		// Set input format class which parses the input HIB and spawns map
		// tasks
		job.setInputFormatClass(ImageBundleInputFormat.class);
		// Set the driver, mapper, and reducer classes which express the
		// computation
		job.setJarByClass(Test.class);
		job.setMapperClass(TestMapper.class);
		job.setReducerClass(TestReducer.class);
		// Set the types for the key/value pairs passed to/from map and reduce
		// layers
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		// Set the input and output paths on the HDFS
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// Execute the MapReduce job and block until it complets
		boolean success = job.waitForCompletion(true);

		// Return success or failure
		return success ? 0 : 1;
	}

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new Test(), args);
		System.exit(0);
	}

}