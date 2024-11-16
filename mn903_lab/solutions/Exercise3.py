# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:11:11 2024

@author: Kieffer
"""

from scipy.io import wavfile
import scipy.io
filename = 'Alarm05.wav'
samplerate, data = wavfile.read(filename)

