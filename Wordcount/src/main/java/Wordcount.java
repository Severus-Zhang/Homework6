import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;


public class Wordcount {
    public static class InvertedMapper extends Mapper<Object, Text,Text, Text> {
        static enum CountersEnum { INPUT_WORDS }
        private Text one = new Text("1");
        private Text word = new Text();
        private Set<String> patternsToSkip = new HashSet<String>(); // 用来保存所有停用词 stopwords
        private Set<String> punctuations = new HashSet<String>(); // 用来保存所有要过滤的标点符号 stopwords

        private Configuration conf;
        private BufferedReader fis; // 保存文件输入流

        @Override
        public void setup(Context context) throws IOException,
                InterruptedException {
            conf = context.getConfiguration();


            if (conf.getBoolean("wordcount.skip.patterns", false)) { // 配置文件中的wordcount.skip.patterns功能是否打开
                URI[] patternsURIs = Job.getInstance(conf).getCacheFiles(); // getCacheFiles()方法可以取出缓存的本地化文件，本例中在main设置

                Path patternsPath = new Path(patternsURIs[0].getPath());
                String patternsFileName = patternsPath.getName();
                parseSkipFile(patternsFileName); // 将文件加入过滤范围，具体逻辑参见parseSkipFile(String fileName)

                Path punctuationsPath = new Path(patternsURIs[1].getPath());
                String punctuationsFileName = punctuationsPath.getName();
                parseSkipPunctuations(punctuationsFileName); // 将文件加入过滤范围，具体逻辑参见parseSkipFile(String fileName)
            }
        }

        private void parseSkipPunctuations(String fileName) {
            try {
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) { // SkipFile的每一行都是一个需要过滤的pattern，例如\!
                    punctuations.add(pattern);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file "
                        + StringUtils.stringifyException(ioe));
            }
        }


        private void parseSkipFile(String fileName) {
            try {
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) { // SkipFile的每一行都是一个需要过滤的pattern，例如\!
                    patternsToSkip.add(pattern);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file "
                        + StringUtils.stringifyException(ioe));
            }
        }


        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit)context.getInputSplit();
            String textName = fileSplit.getPath().getName();
            String line=value.toString().toLowerCase();
            for (String pattern : punctuations) { // 将数据中所有满足patternsToSkip的pattern都过滤掉, 用""代替
                line = line.replaceAll(pattern, " ");
            }
            StringTokenizer itr = new StringTokenizer(line);
            while (itr.hasMoreTokens()) {
                String this_word=itr.nextToken();
                boolean flag=false;
                //判断是否长度小于3
                if(this_word.length()<3) {
                    flag=true;
                }
                //判断是否是数字，用正则表达式
                if(Pattern.compile("^[-\\+]?[\\d]*$").matcher(this_word).matches()) {
                    flag=true;
                }
                //判断是否是停用词
                if(patternsToSkip.contains(this_word)){
                    flag=true;
                }
                if(!flag) {
                    word.set(this_word+":"+textName);
                    context.write(word, one);
                    Counter counter = context.getCounter(
                            CountersEnum.class.getName(),
                            CountersEnum.INPUT_WORDS.toString());
                    counter.increment(1);
                }
            }
        }

    }

    public static class InvertedCombiner extends Reducer<Text, Text, Text, Text> {

        private Text result = new Text();
        private Text combinerOutKey = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values,
                              Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
            int sum =0;
            for(Text val:values){
                sum += Integer.parseInt(val.toString());
            }
            String[] strs = key.toString().split(":");
            combinerOutKey.set(strs[0]+":"+sum);
            result.set(strs[1] + ":" + sum);
            context.write(combinerOutKey, result);
        }
    }

    public static class NewPartitioner extends HashPartitioner<Text,Text>{
        @Override
        public int getPartition(Text key, Text value, int numReduceTasks){
            String[] term = key.toString().split(":");
            return (term[0].hashCode() & 2147483647) % numReduceTasks;
        }
    }

    public static class DecreasingComparator extends Text.Comparator {   //改变默认的排序
        @SuppressWarnings("rawtypes")
        public int compare(Text a, Text b){
            return -super.compare(a, b);
        }
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static class InvertedReducer extends Reducer<Text, Text, Text, Text>{
        private Text result = new Text();
        private Text ReducerOutKey = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
                throws IOException, InterruptedException {
            StringBuffer sb = new StringBuffer();

            for(Text val:values){
                sb.append(val.toString());
                sb.append(",");
            }
            String temp = sb.toString().substring(0,sb.toString().lastIndexOf(","));
            ReducerOutKey.set(temp);
            String[] newkey = key.toString().split(":");
            context.write(new Text(newkey[0]),ReducerOutKey);

        }
    }

    public static class SortMapper extends Mapper<Object, Text, Text, Text>{
        protected void map(Object key, Text value, Mapper<Object, Text, Text, Text>.Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] keyValueStrings = line.split("\t");
            if (keyValueStrings.length != 2) {
                System.err.println("error");
                return;
            }
            String outkey = keyValueStrings[0];
            String outvalue = keyValueStrings[1];
            context.write(new Text(outkey), new Text(outvalue));
        }
    }

    public static class SecondReducer extends Reducer<Text, Text, Text, Text>{
        private Text result = new Text();
        private Text ReducerOutKey = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
                throws IOException, InterruptedException {
            StringBuffer sb = new StringBuffer();

            for(Text val:values){
                sb.append(val.toString());
                sb.append(",");
            }
            String temp = sb.toString().substring(0,sb.toString().lastIndexOf(","));
            ReducerOutKey.set(temp);
            context.write(key,ReducerOutKey);

        }
    }


    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException{
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();
        if ((remainingArgs.length != 2) && (remainingArgs.length != 5)) {
            System.err.println("Usage: wordcount <in> <out> [-skip punctuations skipPatternFile]");
            System.exit(2);
        }

        Job job = new Job(conf, "invertedIndex");
        job.setJarByClass(Wordcount.class);
        job.setMapperClass(InvertedMapper.class);
        job.setInputFormatClass(TextInputFormat.class);


        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setPartitionerClass(NewPartitioner.class);
        job.setSortComparatorClass(DecreasingComparator.class);
        job.setCombinerClass(InvertedCombiner.class);
        job.setReducerClass(InvertedReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        List<String> otherArgs = new ArrayList<String>(); // 除了 -skip 以外的其它参数
        for (int i = 0; i < remainingArgs.length; ++i) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri()); // 将 -skip 后面的参数，即skip模式文件的url，加入本地化缓存中
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true); // 这里设置的wordcount.skip.patterns属性，在mapper中使用
            } else {
                otherArgs.add(remainingArgs[i]); // 将除了 -skip 以外的其它参数加入otherArgs中
            }
        }
        FileInputFormat.addInputPath(job, new Path(otherArgs.get(0)));
        Path fortmp=new Path("fortmp");
        FileOutputFormat.setOutputPath(job, fortmp);
        if(job.waitForCompletion(true))
        {
            //新建一个job处理排序和输出格式
            Job sortJob = Job.getInstance(conf, "sort file");
            sortJob.setJarByClass(Wordcount.class);

            FileInputFormat.addInputPath(sortJob, fortmp);

            //map后交换key和value
            sortJob.setMapperClass(SortMapper.class);
            sortJob.setReducerClass(SecondReducer.class);
            FileOutputFormat.setOutputPath(sortJob, new Path(otherArgs.get(1)));
            sortJob.setOutputKeyClass(Text.class);
            sortJob.setOutputValueClass(Text.class);
            System.exit(sortJob.waitForCompletion(true) ? 0 : 1);
            System.out.println("yes");
        }

    }


}