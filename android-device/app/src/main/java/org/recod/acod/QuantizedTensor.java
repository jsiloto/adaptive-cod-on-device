package org.recod.acod;

import android.graphics.Bitmap;

import org.pytorch.Tensor;

public class QuantizedTensor {
    public float scale = 0.0f;
    public int zero_point = 0;
    public byte [] qx;
    public int originalWidth = 0;
    public int originalHeight = 0;
    public int c, w, h;
    public String mode = "1.0f";
    public String imageId = "";


    public QuantizedTensor(Bitmap bitmap){
        int [] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        qx = new byte[3*pixels.length];
        for(int i=0; i<pixels.length; i++){
            qx[i+2*pixels.length] = (byte) (pixels[i] & 0xFF);
            qx[i+pixels.length] = (byte) ((pixels[i] >> 8) & 0xFF);
            qx[i]= (byte) ((pixels[i] >> 16) & 0xFF);
        }
    }


    public QuantizedTensor(final Tensor outputTensor, int num_bits, String imageId){
        this.imageId = imageId;
        this.c = (int) outputTensor.shape()[1];
        this.w = (int) outputTensor.shape()[2];
        this.h = (int) outputTensor.shape()[3];

        float [] tensor = outputTensor.getDataAsFloatArray();
        num_bits = 8;
        float qmin = 0.0f;
        float qmax = (float) Math.pow(2.0, num_bits - 1.0);
        float inf = Float.POSITIVE_INFINITY;
        float max_val = -inf;
        float min_val = inf;
        for(int i =0; i< tensor.length; i++){
            if(tensor[i] > max_val){ max_val = tensor[i]; }
            if(tensor[i] < min_val){ min_val = tensor[i]; }
        }

        scale = (max_val - min_val) / (qmax - qmin);
        float initial_zero_point = qmin - min_val/scale;
        float zp = initial_zero_point;
        if(initial_zero_point < qmin) { zp = qmin; }
        else if(initial_zero_point > qmax) { zp = qmax; }

        zero_point = (int)zp;
        qx  = new byte[tensor.length];
        for(int i =0; i< tensor.length; i++){
            float a = (zero_point + tensor[i] / scale);
            qx[i] = (byte)Math.round(clamp(a, qmin, qmax));
        }


    }

    public static float clamp(float val, float min, float max) {
        return Math.max(min, Math.min(max, val));
    }

}
