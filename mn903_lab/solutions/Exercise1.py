# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:00:25 2024

@author: Kieffer
"""

import numpy as np

# Zeroth-order entropy evaluation function
def entropy(source):
    alphabet = list(set(source))

    freq = np.zeros(len(alphabet))

    # Evaluate the frequency of each symbol
    for i in range(len(alphabet)):
        freq[i] = source.count(alphabet[i])
        
    freq = freq/len(source)
        
    # Evaluate the entropy
    return -sum(freq*np.log2(freq))

# First-order entropy evaluation function
def entropy1(source):
    alphabet = list(set(source))
    
    # Estimate joint probability of occurence Pr(X_2=k,X_1=l)
    P = np.zeros((len(alphabet),len(alphabet)))
    for i in range(len(source)-1):
        k = alphabet.index(source[i])
        l = alphabet.index(source[i+1])
        P[k,l] += 1
    P = P/len(source)

    # Transition probability matrix Pr(X_2=k|X_1=l)
    T = np.zeros((len(alphabet),len(alphabet)))
    for i in range(len(alphabet)):
        T[i,:] = P[i,:] / sum(P[i,:])
        
    H1 = 0
    for i in range(len(alphabet)):
        H1 += -sum(P[i,:]*np.log2(T[i,:]+1e-10))
    
    # Evaluate the entropy
    return H1


with open("Declaration1789.txt", "r") as file:
    texte = file.read()
    
texte_list = list(texte)

H0 = entropy(texte_list)

print(f'Entropy: {H0:.3f} bits/symb')

H1 = entropy1(texte_list)
print(f'First-order entropy: {H1:.3f} bits/symb')
