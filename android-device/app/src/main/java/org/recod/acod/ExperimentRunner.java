package org.recod.acod;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.pytorch.Tensor;

import java.io.File;


public class ExperimentRunner {
    private int max_images = 500;
    private PytorchModuleWrapper moduleWrapper;
    private ApiHandler apiHandler;

    ExperimentRunner(PytorchModuleWrapper moduleWrapper, ApiHandler apiHandler, Dataset dataset){
        this.moduleWrapper = moduleWrapper;
        this.apiHandler = apiHandler;
    }

    public void run() {
        File[] imageList = Dataset.getInstance().getFileList();
        int max_images = 500;
        
        for (int i = 0; i < max_images; i++) {
            Bitmap bitmap = BitmapFactory.decodeFile(imageList[i].getAbsolutePath());
            Tensor qx = moduleWrapper.run(bitmap);
            System.out.println(String.format("ExperimentOutput: %d images", i+1));
        }
    }
}
