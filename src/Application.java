import hipi.imagebundle.mapreduce.ImageBundleInputFormat;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Application extends Configured implements Tool {

	static {
		// System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.load("/home/thomas/opencv/opencv3/build/lib/libopencv_java300.so");
		System.out.println(System.getenv());
		System.out.println("LIBRARY PATH:"
				+ System.getProperty("java.library.path"));
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
		job.setJarByClass(Application.class);
		job.setMapperClass(OCVCMapper.class);
		job.setReducerClass(OCVCReducer.class);
		// Set the types for the key/value pairs passed to/from map and reduce
		// layers
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
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
		ToolRunner.run(new Application(), args);
		System.exit(0);
	}

}