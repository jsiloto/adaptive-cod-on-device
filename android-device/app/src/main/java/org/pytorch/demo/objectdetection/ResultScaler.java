package org.pytorch.demo.objectdetection;

import android.graphics.Rect;

import java.util.ArrayList;

public class ResultScaler {
    private float imgScaleX, imgScaleY, ivScaleX, ivScaleY, startX, startY;
    public ResultScaler(float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        this.imgScaleX = imgScaleX;
        this.imgScaleY = imgScaleY;
        this.ivScaleX = ivScaleX;
        this.ivScaleY = ivScaleY;
        this.startX = startX;
        this.startY = startY;
    }

    public ArrayList<Result> RescaleResults(ArrayList<Result> original){
        ArrayList<Result> rescaled = new ArrayList<Result>();
        for(Result r: original){
            rescaled.add(new Result(
                    r.classIndex,
                    r.score,
                    new Rect((int)(startX+ivScaleX*r.rect.left),
                            (int)(startY+ivScaleY*r.rect.top),
                            (int)(startX+ivScaleX*r.rect.right),
                            (int)(startY+ivScaleY*r.rect.bottom))));
        }
        return rescaled;
    }

}
