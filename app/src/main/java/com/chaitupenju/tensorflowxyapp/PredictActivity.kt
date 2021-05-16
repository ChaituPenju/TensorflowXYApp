package com.chaitupenju.tensorflowxyapp

import android.os.Bundle
import android.text.TextUtils
import androidx.appcompat.app.AppCompatActivity
import com.chaitupenju.tensorflowxyapp.databinding.ActivityPredictBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

class PredictActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPredictBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // inflate xml using view binding
        binding = ActivityPredictBinding.inflate(layoutInflater)
        // set binding root as content view(setting xml resource throws errors)
        setContentView(binding.root)


        binding.btnPredict.setOnClickListener {
            // edittext validation
            if (TextUtils.isEmpty(binding.etXValue.text.toString())) {
                binding.tilXValue.error = "Input is required!"

                return@setOnClickListener
            }
            binding.tilXValue.error = null

            // get predicted value and set it to textviews
            binding.etXValue.text.toString().toFloat().let { xValue ->
                getPredictedYValue(xValue).let { yValue ->
                    binding.tvPredictedValue.text = String.format(resources.getString(R.string.predicted_value), yValue.toString())
                    binding.tvActualValue.text = String.format(resources.getString(R.string.actual_value), "${2 * xValue - 1}(2 * $xValue - 1)")
                }
            }
        }
    }


    // takes in tflite model file path from assets, the input and predicted output
    private fun getPredictedYValue(xValue: Float): Float {
        // create a model
        val predictXyModel = FileUtil.loadMappedFile(baseContext, "predict_xy_model.tflite")
        // create a interpreter to pass input and output values
        val predictXyInterpreter = Interpreter((predictXyModel) as ByteBuffer)

        // create a float[1][1] variable to store output
        val out = Array(1) { FloatArray(1) }
        // pass input and output values into predictor run
        predictXyInterpreter.run(floatArrayOf((xValue)), out)

        // get the output(predicted) value
        return out[0][0]
    }
}