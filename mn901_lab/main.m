function main()
%% Load data
data_filepath = "eeg1.mat";
data = load(data_filepath, "fs", "s");
fs = data.fs;  % sampling frequency
x = data.s;  % samples
T = length(x);  % #samples

%% Compute coefs & variance with different order p
ps = 1:100;
as = [];
vs = [];
aics = [];

for p = ps
    [a, v] = aryule(x, p);
    aic_p = aic(p, T, v);
    as = [as, a];
    vs = [vs, v];
    aics = [aics, aic_p];
end

%% Find best p
[aic_opt, i_opt] = min(aics);  % optimal index
p_opt = ps(i_opt);
a_opt = as(i_opt);
v_opt = vs(i_opt);
fprintf(['p_optimal=%d\t' ...
    'aic_optimal=%f\n'], p_opt, aic_opt);

% plot
figure();
plot(ps, aics, 'o');
line(ps, aics);
ylim = get(gca, 'YLim');
line([p_opt p_opt], [ylim(1) ylim(2)], 'Color', 'red', 'LineStyle', '--');
xlabel('p');
ylabel('AIC');

%% Spectrum
[spectrum, freq] = pyulear(x, p_opt, T, fs);

% plot
figure();
plot(freq, spectrum);
xlabel('Frequency domain (Hz)');
ylabel('Spectrum S');
