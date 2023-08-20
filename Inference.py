#!/usr/bin/env python

import argparse
import onnxruntime as ort
from transformers import Pix2StructProcessor, AutoTokenizer
from PIL import Image
import numpy as np
import time

def load_onnx_model(model_path):
    return ort.InferenceSession(model_path)

def main():
    parser = argparse.ArgumentParser(description="Run ONNX model for inference")
    parser.add_argument("-m", "--model", required=True, help="Path to the ONNX model folder")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    parser.add_argument("-q", "--question", required=True, help="Question text for inference")
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    question = args.question

    try:
        ort_sess = load_onnx_model(model_path)
        image = Image.open(image_path)
        input_names = [input_.name for input_ in ort_sess.get_inputs()]
        output_names = [output.name for output in ort_sess.get_outputs()]
        
        HF_MODEL_NAME = "google/pix2struct-docvqa-base"

        processor = Pix2StructProcessor.from_pretrained(HF_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

        encoded_input = processor(images=image, text=question, return_tensors="np")
        encoded_text = tokenizer(question, return_tensors="np")

        onnx_time = time.time()

        onnx_predictions = ort_sess.run(output_names, 
                                        input_feed={
                                            input_names[0]: encoded_input["flattened_patches"], 
                                            input_names[1]: encoded_input["attention_mask"].astype(np.int64), 
                                            input_names[2]: encoded_text["input_ids"].astype(np.int64)
                                        })
        decoded_predictions = processor.decode(onnx_predictions[0][0].argmax(axis=1), skip_special_tokens=True).strip()

        print("Inference result:", decoded_predictions)
        print("Time Taken:", round(time.time() - onnx_time, 3))

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
