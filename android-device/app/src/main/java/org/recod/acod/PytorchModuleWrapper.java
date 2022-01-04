package org.recod.acod;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class PytorchModuleWrapper {
    Module mModule;
    boolean dummy = false;

    public PytorchModuleWrapper(String modulePath) {
        mModule = LiteModuleLoader.load(modulePath);
    }

    public PytorchModuleWrapper(String modulePath, boolean dummy) {
        mModule = LiteModuleLoader.load(modulePath);
        this.dummy = dummy;
    }


    public QuantizedTensor run(Bitmap bitmap, String imageId) {
        Tensor outputTensor = null;
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        if(dummy){
            outputTensor = inputTensor;
        }
        if (!dummy) {
            IValue outputTuple = mModule.forward(IValue.from(inputTensor));
            outputTensor = outputTuple.toTensor();
        }
        QuantizedTensor qx = new QuantizedTensor(outputTensor, 8, imageId);
        qx.originalWidth = bitmap.getWidth();
        qx.originalHeight = bitmap.getHeight();
        return qx;
    }

    public void setWidth(float width) {
        if(!dummy){
            mModule.runMethod("set_width", IValue.from(width));
        }
    }
}
