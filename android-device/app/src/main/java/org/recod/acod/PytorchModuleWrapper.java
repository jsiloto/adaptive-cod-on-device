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
    float width = 1.0f;

    public PytorchModuleWrapper(String modulePath) {
        mModule = LiteModuleLoader.load(modulePath);
    }

    public PytorchModuleWrapper(String modulePath, boolean dummy) {
        this.dummy = dummy;
        if(!dummy){
            mModule = LiteModuleLoader.load(modulePath);
        }
    }


    public QuantizedTensor run(Bitmap bitmap, String imageId) {
        Tensor outputTensor = null;
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        if(dummy){
            SystemClock.sleep(500);
            outputTensor = inputTensor;
        }
        if (!dummy) {
            IValue outputTuple = mModule.forward(IValue.from(inputTensor));
            outputTensor = outputTuple.toTensor();
        }
        QuantizedTensor qx = new QuantizedTensor(outputTensor, 8, imageId);
        qx.originalWidth = bitmap.getWidth();
        qx.originalHeight = bitmap.getHeight();
        qx.alpha = this.width;
        return qx;
    }

    public void setWidth(float width) {
        if(!dummy){
            this.width = width;
            mModule.runMethod("set_width", IValue.from(width));
        }
    }
}
