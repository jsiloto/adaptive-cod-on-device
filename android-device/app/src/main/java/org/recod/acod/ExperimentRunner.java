package org.recod.acod;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.pytorch.Tensor;

import java.io.File;


public class ExperimentRunner {
    private int max_images = 100;
    private PytorchModuleWrapper moduleWrapper;
    private ApiHandler apiHandler;
    private File[] imageList = Dataset.getInstance().getFileList();

    ExperimentRunner(PytorchModuleWrapper moduleWrapper, ApiHandler apiHandler, Dataset dataset){
        this.moduleWrapper = moduleWrapper;
        this.apiHandler = apiHandler;
    }

    public void run() {
        int i = 0;
        while(true){
            i++;
            Bitmap bitmap = BitmapFactory.decodeFile(imageList[i%max_images].getAbsolutePath());
            Tensor qx = moduleWrapper.run(bitmap);
            Log.d("MyTAG", String.format("ExperimentOutput: %d images", i+1));
//            System.out.println(String.format("ExperimentOutput: %d images", i+1));
        }
    }
}
