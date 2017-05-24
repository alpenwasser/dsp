clear all;close all;clc

Fs = 1000; % Sampling Frequency in Hertz
Ts = 1/Fs; % Sampling Period
L  = 1e4;  % Length of signal vector
t  = (0:L-1)*Ts; % time vector

signal = 0;
for k = 1:20
    component = 4/pi*1/(2*k-1) * sin(2*pi*(2*k-1)*t);
    signal = signal + component;
end
figure;
plot(t,signal)

frequencyVector = fft(signal);
frequenciesShifted = fftshift(frequencyVector);
specTwoSide = abs(frequencyVector/L); % two-sided spectrum
specOneSide = specTwoSide(1:L/2+1); % one-sided spectrum
specOneSide(2:end-1) = 2*specOneSide(2:end-1);

f = Fs*(0:(L/2))/L;

figure;
plot(f,specOneSide);