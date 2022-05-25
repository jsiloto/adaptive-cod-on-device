package org.recod.acod;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class PytorchModuleWrapper {
    Module mModule;
    boolean dummy = false;
    int mode = 1;

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

    public void setMode(int mode) {
        if (!dummy) {
            this.mode = mode;
            IValue input = IValue.from(mode);
            Log.d("MyTAG", input.toString());
            mModule.runMethod("set_mode", input);
//            mModule.runMethod("test");


//            mModule.runMethod("set_mode", input);
            try {
                //            mModule.runMethod("set_config", IValue.from(width), IValue.from(width), IValue.from((int)(4*width)));
//                mModule.runMethod("set_width", IValue.from(0.25));
//                int i = 1 + (int) ((width - 0.25) * 12);
//                i = 1 +(int)((mode-0.25)*4);
//                IValue input = IValue.from((long)(mode));
//                Log.d("MyTAG", input.toString());
                mModule.runMethod("set_mode", input);
            } catch (Exception e) {
            }
        }
    }
}
