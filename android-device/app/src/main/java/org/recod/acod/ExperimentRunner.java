package org.recod.acod;

import static androidx.core.content.ContextCompat.getSystemService;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.PowerManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


public class ExperimentRunner {
    private int max_images = 500;
    private PytorchModuleWrapper moduleWrapper;
    private ApiHandler apiHandler;
    private File[] imageList;
    private PowerManager.WakeLock wakeLock;

    ExperimentRunner(PytorchModuleWrapper moduleWrapper, ApiHandler apiHandler, Dataset dataset){
        this.moduleWrapper = moduleWrapper;
        this.apiHandler = apiHandler;
        this.imageList = dataset.getFileList(max_images);
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyApp::MyWakelockTag");
    }

    public void run() {
        wakeLock.acquire();
        for (int i = 0; i < max_images; i++) {
//            final Runtime runtime = Runtime.getRuntime();
//            final long usedMemInKB=(runtime.totalMemory() - runtime.freeMemory()) / 1024L;
//            System.out.println(usedMemInKB);

            String imageId = imageList[i].getName();
            try {
                FileInputStream stream = new FileInputStream(imageList[i]);
                Bitmap bitmap = BitmapFactory.decodeStream(stream);
                QuantizedTensor qx = moduleWrapper.run(bitmap, imageId);
//                apiHandler.postSplitTensor(qx, null);
                System.out.println(String.format("ExperimentOutput: %d images", i+1));
                stream.close();
            } catch (IOException e) {
                System.out.println("Error processing tensor");
                e.printStackTrace();
            }
        }
        wakeLock.release();
    }
}
