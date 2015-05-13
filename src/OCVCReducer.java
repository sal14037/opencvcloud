import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class OCVCReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

	public void reduce(IntWritable key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
		int total = 0;
		for (Text val : values) {
			// Emit output of job which will be written to HDFS
			context.write(key, new Text(key + " " + val));
			total++;
		}
	}
}
