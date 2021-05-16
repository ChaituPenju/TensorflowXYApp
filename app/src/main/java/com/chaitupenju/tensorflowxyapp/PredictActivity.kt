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
        binding = ActivityPredictBinding.inflate(layoutInflater)
        setContentView(binding.root)


        binding.btnPredict.setOnClickListener {
            if (TextUtils.isEmpty(binding.etXValue.text.toString())) {
                binding.tilXValue.error = "Input is required!"

                return@setOnClickListener
            }
            binding.tilXValue.error = null

            binding.etXValue.text.toString().toFloat().let { xValue ->
                getPredictedYValue(xValue).let { yValue ->
                    binding.tvPredictedValue.text = String.format(resources.getString(R.string.predicted_value), yValue.toString())
                    binding.tvActualValue.text = String.format(resources.getString(R.string.actual_value), "${2 * xValue - 1}(2 * $xValue - 1)")
                }
            }
        }
    }

    private fun getPredictedYValue(xValue: Float): Float {
        val predictXyModel = FileUtil.loadMappedFile(baseContext, "predict_xy_model.tflite")
        val predictXyInterpreter = Interpreter((predictXyModel) as ByteBuffer)

        val out = Array(1) { FloatArray(1) }
        predictXyInterpreter.run(floatArrayOf((xValue)), out)

        return out[0][0]
    }
}