## Barebones python script to run inference coreml models on M1 hardware

Production machines are usually large beefy rigs, running windows on intel CPUs and sporting fast NVidia cards. On the other hand, my day to day machine is a first generation apple M1 which continues to impress me years after it came out. Fast, efficient, cheap and elegant... Unfortunately, its biggest issue is the inability to run CUDA on it, and all the AI models are written for CUDA.

Apple has come out with coremltools to convert the main families of models and a few ready to use translated models. This script runs some of those models in python nin the quickest simplest way. It's the missing quickstart.

The available apple translated models can be downloaded from their web site: https://developer.apple.com/machine-learning/models/

## SAM2, using GPU on apple silicon

The SAM2_1 models converted to coreml are available on hugginface:
https://huggingface.co/collections/apple/core-ml-segment-anything-2-66e4571a7234dc2560c3db26

You need to compile the .mlpackage files to .mlmodelc and put them in the modelsC directory.

The script requires 3 options: an image, a list of (x,y) tuples representing the prompt points, and a correcponding list of integers to speficy if you want to add or remove an area from a mask.
python sam2-createmask.py --image dogs.jpg --points "[(0.5,0.65),(.5,.55),(.1,.1)]" --labels "[1,1,0]"

