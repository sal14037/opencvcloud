import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class OCVCReducer extends Reducer<Text, IntWritable, IntWritable, Text> {

	public void reduce(IntWritable key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
		int total = 0;
		String result = " Detected faces: ";
		for (Text val : values) {
			// Emit output of job which will be written to HDFS
			context.write(key, new Text(key + result + val));
			total++;
		}
	}
}
