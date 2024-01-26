import os
import logging
import json
import base64
import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    global model, processor

    model_name = "openai/whisper-large-v2"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()    


# Run the PyTorch model, for functional and performance comparison
def run(payload):

    logging.info("Running inference ...")
    data = json.loads(payload)
    audio = base64.b64decode(data["audio"])
    with open("temp.mp3", "wb") as f:
        f.write(audio)

    speech, _ = librosa.load("temp.mp3")

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features, max_length=500)

    return processor.batch_decode(predicted_ids, skip_special_tokens = True)

if __name__ == '__main__':
    init()

    audio_file = "../audio.mp3"

    with open(audio_file, "rb") as f:
        audio = f.read()

    audio_string = base64.b64encode(audio).decode("utf-8")    

    payload = json.dumps({"audio": audio_string})

    print(run(payload))


