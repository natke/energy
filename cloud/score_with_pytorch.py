import os
import logging
import json
import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import onnxruntime
from onnxruntime_extensions import get_library_path

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    global session, model, processor

    model_name = "openai/whisper-tiny.en"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    if model_dir == None:
        model_dir = "./"
    model_path = os.path.join(model_dir, model_name + ".onnx")

    # Create an ONNX Runtime session to run the ONNX model
    options = onnxruntime.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    session = onnxruntime.InferenceSession(model_path, options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    


# Run the PyTorch model, for functional and performance comparison
def run_pytorch(audio):

    model.eval()    

    # TODO Work out why the audio is not being loaded correctly
    speech, _ = librosa.load(audio)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features, max_length=500)

    return processor.batch_decode(predicted_ids, skip_special_tokens = True)

# Run the ONNX model with ONNX Runtime
def run(raw_data):

    inputs = json.loads(raw_data)

    inputs = {
        "audio_stream": np.array([inputs["audio"]], dtype=np.uint8),
        "max_length": np.array([500], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "num_beams": np.array([2], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
        "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
    }

    # TODO Work out why the output shape is not correct
    outputs = session.run(None, inputs)[0]

    return outputs


if __name__ == '__main__':
    init()

    audio_file = "audio.mp3"

    with open(audio_file, "rb") as f:
        audio = np.asarray(list(f.read()), dtype=np.uint8)

    raw_data = json.dumps({"audio": audio.tolist()})

    print(run(raw_data))

    print(run_pytorch(audio_file))


