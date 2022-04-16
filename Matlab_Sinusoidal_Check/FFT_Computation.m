Fs = (1/0.1); % Sampling frequency
t = 0:1/Fs:540*(1/Fs); % Time vector of 1 second
x = Actual(:,1);
nfft = 541; % Length of FFT
% Take fft, padding with zeros so that length(X)is equal to nfft
X = fft(x,nfft);
%FFT is symmetric, throw away half
X = X(1:nfft/2);
% Take the magnitude of fft of x
mx_gp = abs(X);
% Frequency vector
f = (0:nfft/2-1)*Fs/nfft;
% Generate the plot, title and labels.
plot(f.*2*pi,mx_gp,'b');
