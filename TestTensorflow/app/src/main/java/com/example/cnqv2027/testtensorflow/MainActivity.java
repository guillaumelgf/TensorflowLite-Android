package com.example.cnqv2027.testtensorflow;

import android.content.res.AssetFileDescriptor;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private final String MODEL_NAME = "model2.tflite";

    private final String defaut = "Click on predict to obtain a result.";
    private Button predictButton = null;
    private Button razButton = null;
    private EditText sepalLengthText = null;
    private EditText sepalWidthText = null;
    private EditText petalLengthText = null;
    private EditText petalWidthText = null;
    private TextView result = null;

    protected Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Set the buttons
        predictButton = (Button) findViewById(R.id.predict);
        razButton = (Button) findViewById(R.id.raz);
        predictButton.setOnClickListener(predictListener);
        razButton.setOnClickListener(razListener);

        //Set the texts
        sepalLengthText = (EditText) findViewById(R.id.sepalLength);
        sepalWidthText = (EditText) findViewById(R.id.sepalWidth);
        petalLengthText = (EditText) findViewById(R.id.petalLength);
        petalWidthText = (EditText) findViewById(R.id.petalWidth);

        result = (TextView) findViewById(R.id.result);

        // First we must create the tflite object, loaded from the model file
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }


    private View.OnClickListener predictListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            float sepalLength = Float.valueOf(sepalLengthText.getText().toString());
            float sepalWidth = Float.valueOf(sepalWidthText.getText().toString());
            float petalLength = Float.valueOf(petalLengthText.getText().toString());
            float petalWidth = Float.valueOf(petalWidthText.getText().toString());

            float[][] prediction = doInference(sepalLength, sepalWidth, petalLength, petalWidth);
            String predictionText = String.format("Iris_setosa: %.2f%% \nIris-versicolor: %.2f%% \nIris-virginica: %.2f%%", prediction[0][0]*100, prediction[0][1]*100, prediction[0][2]*100);
            result.setText(predictionText);
        }
    };

    private View.OnClickListener razListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            sepalLengthText.getText().clear();
            sepalWidthText.getText().clear();
            petalLengthText.getText().clear();
            petalWidthText.getText().clear();

            result.setText(defaut);
        }
    };

    /* Memory-map the model in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        // Open the model using an input stream, and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public float[][] doInference(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
        float[][] inputVal = new float[1][4];
        float[][] outputVal = new float[1][3];

        inputVal[0][0] = sepalLength;
        inputVal[0][1] = sepalWidth;
        inputVal[0][2] = petalLength;
        inputVal[0][3] = petalWidth;

        tflite.run(inputVal, outputVal);
        return outputVal;
    }
}
