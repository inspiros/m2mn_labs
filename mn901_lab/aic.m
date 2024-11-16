function [res] = aic(p, T, v)
res = 2 * p / T + log(sqrt(v));
