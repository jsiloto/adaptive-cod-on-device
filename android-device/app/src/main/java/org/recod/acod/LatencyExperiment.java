package org.recod.acod;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;

import androidx.annotation.ColorInt;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.stream.Stream;


public class LatencyExperiment  {

    private Module module = null;
    private String modelName;

    LatencyExperiment(String modelName, Context context) throws IOException {
        String modulePath = Helper.assetFilePath(context, modelName);
        this.modelName = modelName;
        module = LiteModuleLoader.load(modulePath);
        try {
            module.runMethod("set_width", IValue.from(0.25));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String run(){

        int max_images = 5;
        File[] imageList = Dataset.getInstance().getFileList(max_images);
        Bitmap[] bitmaps = Stream.of(imageList).map((a)->{
            FileInputStream stream = null;
            try {
                stream = new FileInputStream(a);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            Bitmap bitmap = BitmapFactory.decodeStream(stream);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 768, 768, true);
            return resizedBitmap;
        }).toArray(Bitmap[]::new);

        int n = 2;
        double time = 0;
        String output = this.modelName;
        /////////////////////////// Warmup //////////////////////////////
        double average = 10000.0;
        double alpha = 0.25;
        while( Math.abs(time-average)/time > 0.10){
            time = 0;
            for(int i=0; i<n; i++){
                time += (runSingle(bitmaps));
            }
            time = time/n;
            average = alpha*average + (1-alpha)*time;
            System.out.println(String.format("now:%3.2f, avg:%3.2f", time, average));
        }

        output+= String.format("\nWarmup: %2.2f ms", time);

        /////////////////////////// Experiment //////////////////////////////
        time = 0;
        for(int i=0; i<n; i++){
            time += runSingle(bitmaps);
        }
        time = time/n;
        output+= String.format("\nExperiment: %3.2f ms", time);

        return output;
    }

    public double runSingle(Bitmap[] bitmaps){
        int numImages = bitmaps.length;
        long tStart = System.currentTimeMillis();
        for (Bitmap bitmap: bitmaps) {
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
            IValue outputTuple = module.forward(IValue.from(inputTensor));
//            final Tensor outputTensor = outputTuple.toTensor();
        }
        long tEnd = System.currentTimeMillis();
        double elapsed = (tEnd-tStart)/(numImages*1.0);
        System.out.println(String.format("Single:%3.2f", elapsed));
        return elapsed;
    }


    /**
     * Create a bitmap of specific size with a specific color
     * http://www.java2s.com/example/android/graphics/create-a-bitmap-of-specific-size-with-a-random-color.html
     *
     * @param w - width
     * @param h - height
     * @param color - color integer (not resource id)
     * @return - bitmap of random(ish) color
     */
    public static Bitmap createTestBitmap(int w, int h,
                                          @ColorInt Integer color) {
        Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);

        if (color == null) {
            int colors[] = new int[] { Color.BLUE, Color.GREEN, Color.RED,
                    Color.YELLOW, Color.WHITE };
            Random rgen = new Random();
            color = colors[rgen.nextInt(colors.length - 1)];
        }

        canvas.drawColor(color);
        return bitmap;
    }

}
