package org.recod.acod;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class PytorchModuleWrapper {
    Module mModule;

    public PytorchModuleWrapper(String modulePath) {
        mModule = LiteModuleLoader.load(modulePath);
    }

    public QuantizedTensor run(Bitmap bitmap, String imageId) {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        IValue outputTuple = mModule.forward(IValue.from(inputTensor));
        final Tensor outputTensor = outputTuple.toTensor();

        QuantizedTensor qx = new QuantizedTensor(outputTensor, 8, imageId);
        qx.originalWidth = bitmap.getWidth();
        qx.originalHeight = bitmap.getHeight();
        return qx;
    }
    public void setWidth(float width){
        mModule.runMethod("set_width", IValue.from(width));
    }


}
