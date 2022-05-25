package org.recod.acod;

import android.graphics.Bitmap;
import android.os.SystemClock;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class PytorchModuleWrapper {
    Module mModule;
    boolean dummy = false;
    String mode = "1.0f";

    public PytorchModuleWrapper(String modulePath) {
        mModule = LiteModuleLoader.load(modulePath);
    }

    public PytorchModuleWrapper(String modulePath, boolean dummy) {
        this.dummy = dummy;
        if (!dummy) {
            mModule = LiteModuleLoader.load(modulePath);
        }
    }

    public Tensor run(Bitmap bitmap){
        Tensor outputTensor = null;
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        if (dummy) {
            SystemClock.sleep(500);
            outputTensor = inputTensor;
        }
        if (!dummy) {
            IValue outputTuple = mModule.forward(IValue.from(inputTensor));
            outputTensor = outputTuple.toTensor();
        }
        return outputTensor;
    }


    public QuantizedTensor runQuantized(Bitmap bitmap, String imageId) {
        Tensor outputTensor = run(bitmap);
        QuantizedTensor qx = new QuantizedTensor(outputTensor, 8, imageId);
        qx.originalWidth = bitmap.getWidth();
        qx.originalHeight = bitmap.getHeight();
        qx.mode = this.mode;
        return qx;
    }

    public void setMode(String mode) {
        if (!dummy) {
            this.mode = mode;
            try {
                //            mModule.runMethod("set_config", IValue.from(width), IValue.from(width), IValue.from((int)(4*width)));
//                mModule.runMethod("set_width", IValue.from(0.25));
//                int i = 1 + (int) ((width - 0.25) * 12);
//                i = 1 +(int)((mode-0.25)*4);
                mModule.runMethod("set_mode", IValue.from(mode));
            } catch (Exception e) {
            }
        }
    }
}
