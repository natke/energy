import json
import os
import time
import numpy as np
import onnxruntime
import onnxruntime_extensions as extensions
import onnxruntime_genai as og

def init():

  global model, tokenizer, detokenizer

  device_type = og.DeviceType.CPU
  name = "microsoft/phi-2/cpu-int4"

  # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
  # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
  model_dir = os.getenv('AZUREML_MODEL_DIR')

  model=og.Model(f'{model_dir}/{name}', device_type)

  # Create the tokenizer and detokenizer (temporary, until tokenization API is added to extensions)
  options = onnxruntime.SessionOptions()
  options.register_custom_ops_library(extensions.get_library_path())

  tokenizer = onnxruntime.InferenceSession(f'{model_dir}/microsoft/phi-2/tokenizer.onnx', options)
  detokenizer = onnxruntime.InferenceSession(f'{model_dir}/microsoft/phi-2/detokenizer.onnx', options)

def run(payload):
  
  data = json.loads(payload)

  input = tokenizer.run(None, { "input_text": np.array([data["prompt"]] ) })

  params=og.SearchParams(model)
  params.max_length = 200
  params.input_ids = input[0]

  start_time=time.time()
  output_tokens=model.Generate(params)
  run_time=time.time()-start_time

  output = detokenizer.run(None, { "ids": output_tokens })

  results = {}
  results["output"] = output[0][0]

  return results


if __name__ == '__main__':
    init()

    prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

    payload = json.dumps({"prompt": prompt})

    print(run(payload)["output"])
