/*
	Copyright (C) 2016 - 2019 Pinard Liu(liujianping-ok@163.com)

	https://www.cnblogs.com/pinard

	Permission given to modify the code as long as you keep this declaration at the top

	tensorflow机器学习模型的跨平台上线 https://www.cnblogs.com/pinard/p/9251296.html
*/

import org.tensorflow.*;
import org.tensorflow.Graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;


/**
 * Created by 刘建平pinard on 2018/7/1.
 */
public class TFjavaDemo {
    public static void main(String args[]){
        byte[] graphDef = loadTensorflowModel("D:/rf.pb");
        float inputs[][] = new float[4][6];
        for(int i = 0; i< 4; i++){
            for(int j =0; j< 6;j++){
                if(i<2) {
                    inputs[i][j] = 2 * i - 5 * j - 6;
                }
                else{
                    inputs[i][j] = 2 * i + 5 * j - 6;
                }
            }
        }
        Tensor<Float> input = covertArrayToTensor(inputs);
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session s = new Session(g);
        Tensor result = s.runner().feed("input", input).fetch("output").run().get(0);

        long[] rshape = result.shape();
        int rs = (int) rshape[0];
        long realResult[] = new long[rs];
        result.copyTo(realResult);

        for(long a: realResult ) {
            System.out.println(a);
        }
    }
    static private byte[] loadTensorflowModel(String path){
        try {
            return Files.readAllBytes(Paths.get(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    static private Tensor<Float> covertArrayToTensor(float inputs[][]){
        return Tensors.create(inputs);
    }
}
