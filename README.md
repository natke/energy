# ONNX Runtime energy consumption

UNDER DEVELOPMENT

- [x] Deploy  phi-2
- [ ] Deploy llama
- [ ] Deploy whisper
- [ ] Deploy stable diffusion
- [x] Run on GPU
- [x] Add GPU energy metric to endpoint
- [ ] Mimic a typical user flow
- [ ] Compare the two

## Cloud experimentation

See [./cloud] folder

### Dependencies

See [env.yml](env.yml) in each model directory

To build onnxruntime

```bash
./build.sh --use_cuda --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --build_wheel
```

### Compute


### Model



### Endpoint

```
export ENDPOINT_NAME=phi-2
az ml online-endpoint create --local -n $ENDPOINT_NAME -f endpoint.yml
```

### Deployment

 az ml online-deployment create --local -n blue --endpoint $ENDPOINT_NAME -f deploy-local.yml


```bash
https://whisper-onnx.australiaeast.inference.ml.azure.com/score
```

### Metrics

Go to endpoint in AML

Click on metrics

Add metric GPU energy in Joules

### Scoring scripts

```bash
python score.py
python score_with_pytorch.py
```

### Call scoring endpoint

python call_scoring_endpoint.py
python call_pytorch_endpoint.py

### Measurement tools

https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/dashboard/arm/subscriptions/ea482afa-3a32-437c-aa10-7de928a9e793/resourcegroups/dashboards/providers/microsoft.portal/dashboards/b9f4a0f0-3f2f-479a-9aee-b60085c50077 

## Local experimentation

### Setup

```bash
pip install onnx
pip install transformers
pip install torch (or GPU version)
pip install onnxruntime-gpu
pip install experiment-impact-tracker 
```

#### Windows

Install Power Gadget: https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html

TODO: find Python package for Power Gadget 

### Export model

```bash
python export.py
```

### Run on Windows

In one terminal

```bash
python create_session.py
```

In another

```bash
"c:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe" -file power.csv -cmd python run_session.py
```

### Run on Mac

```bash
python score.py
```

### Reference

https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html 

https://github.com/Breakend/experiment-impact-tracker

nvidia-smi

nvcc

https://github.com/Syllo/nvtop

https://developer.nvidia.com/nvidia-management-library-nvml 

https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli 

https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu 

https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series

![Azure VM NC series](image.png)
