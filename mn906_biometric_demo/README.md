### MN906 - Biometric Security Face Verification Demo

This repo is a minorly modified version of the original code provided by *Prof. D. Petrovska*.
The changes include fixing some bugs, adding new functionalities like landmark visualization, and other optimizations.

#### Requirements

- Python 3.7+
- `numpy`, `scipy`, `opencv-python`, and `dlib`

#### Installation

First, please download the pretrained weights (I don't have the link) and put them in the `models` folder.

##### Using Anaconda

This is self-explainatory.

##### Using `pip`

Without Anaconda, you may encounter issues when trying to install `dlib`.
There are two simple solutions:

- Install a C++ compiler (`MSVC` in Windows and `gcc` in Linux) before running `pip install dlib`.
- Download the appropriate precompiled wheel from https://github.com/z-mahmud22/Dlib_Windows_Python3.x and run `pip install [file_name].whl`.

#### Usage

You can modify the face detection/verification modules' parameters in the file [`parameters.py`](parameters.py).
Then, execute this command:

```
python demo-bio.py
```

During execution, these keys are listened:

- `[space]` for manually enrolling the current detected face (if there is exactly one).
- `s` for saving the current frame as well as crops of detected faces.
- `q` for terminating the demo.

The outputs can be found in a newly created `logs` folder.
