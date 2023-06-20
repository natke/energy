#!/bin/bash

# Set account and subscription
az account set --subscription <subscription ID>
az configure --defaults workspace=<Azure Machine Learning workspace name> group=<resource group>

export ENDPOINT_NAME="whisper-onnx"



