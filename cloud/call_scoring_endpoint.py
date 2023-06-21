import requests
import json
import numpy

audio_file = "audio.mp3"

with open(audio_file, "rb") as f:
    audio = numpy.asarray(list(f.read()), dtype=numpy.uint8)

payload = json.dumps({"audio": audio.tolist()})

response = requests.post("http://localhost:5001/score", data=payload)

print(response.json())