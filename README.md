# Detecting Stable Diffusion Generated Images Using Frequency Artifacts: A Case Study on Disney-Style Art

[![Live Demo](https://img.shields.io/badge/demo-live-blue)](https://zjbthomas.github.io/FreqAIDetector/)


This is the repository for paper [Detecting Stable Diffusion Generated Images Using Frequency Artifacts: A Case Study on Disney-Style Art](https://ieeexplore.ieee.org/abstract/document/10221905) accepted to ICIP 2023.

![network structure](./github/network.png)

## Datasets

The dataset we constructed in Section 3.1 of our paper can be downloaded [here](https://www.dropbox.com/scl/fi/vyk9bi7df6hp46md6mkxg/DisneyDataset.zip?rlkey=4wrahu4usf6g372lhyhdh8bmj&st=wnia1liw&dl=0).

## Checkpoints

Our trained model can be found [here](./checkpoints/ours.pth).

## ONNX

An ONNX version of our trained model can be found [here](./deploy/AWS/Lambda/model.onnx), with inference script [here](./deploy/onnx/onnx_infer.py).

## Model Context Protocol (MCP)

An MCP Server is implemented [here](./deploy/MCP/) as a *Tool* that allows LLMs to analyze images and determine if they are AI-generated.

- MCP Server URL: https://175.178.11.87/freqaidetector-mcp
- Transport: Streamable HTTP
- Authentication: No

> **Note:** ChatGPT currently cannot connect to this MCP Server because it does not support IP-based endpoints; a domain-hosted version compatible with ChatGPT connectors is under development.