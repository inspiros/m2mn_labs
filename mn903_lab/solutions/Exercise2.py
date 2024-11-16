# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:34:13 2024

@author: Kieffer
"""

import numpy as np
import matplotlib.pyplot as plt

def quant_midrise(x,Delta):
    idx = np.ceil(x/Delta)
    qx = idx*Delta - Delta/2

    return (idx,qx)    

def quant_midtread(x,Delta):
    idx = np.ceil((x- Delta/2)/Delta)
    qx = idx*Delta

    return (idx,qx)    

def entropy(source):
    alphabet = list(set(source))
    source_list = list(source)

    freq = np.zeros(len(alphabet))

    # Evaluate the frequency of each symbol
    for i in range(len(alphabet)):
        freq[i] = source_list.count(alphabet[i])
        
    freq = freq/len(source)
        
    # Evaluate the entropy
    return -sum(freq*np.log2(freq))


N = 10000

# Plot of the characteristics of the quantizers
Delta = 0.1
x = np.linspace(-5,5,N)

(idx,qx) = quant_midrise(x, Delta)

plt.plot(x,qx)
plt.grid()
plt.title("Midrise quantizer")
plt.show()

(idx,qx) = quant_midtread(x, Delta)

plt.plot(x,qx)
plt.grid()
plt.title("Midtread quantizer")
plt.show()

# Rate-distorsion evaluation
x = np.random.normal(0,1,N)
(idx,qx) = quant_midrise(x, Delta)

# Evaluation of entropy of quantized version of x
H = entropy(idx)

# Evaluation of distorsion
D = np.var(x-qx)

print(f"Entropy {H:2.3} bits/symb; Distorsion {D:2.5}")


# Rate-distorsion characteristic
Delta = np.logspace(-2, 1,20)

Hmr = []
Dmr = []
Hmt = []
Dmt = []

for i in range(len(Delta)):
    
    (idx,qx) = quant_midrise(x, Delta[i])
    Hmr.append(entropy(idx))
    Dmr.append(np.var(x-qx))
    
    (idx,qx) = quant_midtread(x, Delta[i])
    Hmt.append(entropy(idx))
    Dmt.append(np.var(x-qx))
    
plt.semilogy(Hmr,Dmr)
plt.semilogy(Hmt,Dmt)
plt.legend(("Midrise","Midtread"))
plt.grid()
plt.xlabel("Entropy (bits/symb)")
plt.ylabel("Distorsion")
plt.title("Midrise quantizer")

plt.show()