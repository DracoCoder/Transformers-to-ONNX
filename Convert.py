#!/usr/bin/env python
# Install necessary packages
import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "optimum[onnxruntime]"])




def main():
    try:
        from pathlib import Path
        from optimum.exporters import TasksManager
        from optimum.exporters.onnx import export
        from transformers import Pix2StructForConditionalGeneration

        HF_MODEL_NAME = "google/pix2struct-docvqa-base"
        base_model = Pix2StructForConditionalGeneration.from_pretrained(HF_MODEL_NAME)

        onnx_path = Path(f"./Model/{HF_MODEL_NAME.split("/")[-1]}.onnx")
        onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", base_model, task='visual-question-answering')
        onnx_config = onnx_config_constructor(base_model.config)

        onnx_inputs, onnx_outputs = export(base_model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)

        print("ONNX export completed. Model saved at:", onnx_path)

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
