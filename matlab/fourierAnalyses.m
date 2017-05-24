clear all;close all;clc;
set(0,'DefaultFigureWindowStyle','docked');

% **************************************************************************** %
% CONSTRUCTION OF "ANALOG" INPUT SIGNAL                                        %    
% **************************************************************************** %
% In this case a signal with a high number of samples
% Square wave signal with K harmonics

tmax = 8*pi;        % stop time in seconds
Na = 1e4;           % number of "analog" samples
t = 0:tmax/Na:tmax; % time axis
K = 20;             % number of harmonics
f = 1;              % frequency in Hertz
x = 0;
% for k = 1:K
%     component = 4/pi*1/(2*k-1) * sin(2*pi*(2*k-1)*f*t);
%     x = x + component;
% end
x = sin(2*pi*f*t);
xs = 0;
Fs = 2.1*f;
Ts = 1/Fs;
Nd = floor(tmax/Ts); % Number of digital samples

% Sample the signal
xs = [];
td = 0:tmax/Nd:tmax-tmax/Nd;
tic
for m = 1:Nd
    n = floor(m*Ts*Na/tmax); % Index to sample from x
    xs = [xs x(n)];
end
toc

% Reconstruct the signal
xr = zeros(1,Na);
tic
for k = 1:Na
    for m = 1:Nd
        xr(k) = xr(k) + xs(m)*sinc((k*tmax/Na - m*Ts)/Ts);
    end
end
toc


figure('name','Sampling and Reconstruction');
subplot(3,1,1);
plot(t,x);grid on;axis([0 tmax -1.1 1.1]);title('Source Signal');
subplot(3,1,2);
stem(td,xs);grid on;axis([0 tmax -1.1 1.1]);title('Sample Values');
subplot(3,1,3);
plot(t(1:Na),xr);grid on;axis([0 tmax -1.1 1.1]);title('Reconstructed Signal');

% Discrete Fourier Transform of input signal
X = zeros(1,Nd);
for m = 0:Nd-1
    for n = 0:Nd-1
        X(m+1) = X(m+1) + xs(n+1) * exp(-j*2*pi*m*n/Nd);
    end
end
figure;
subplot(2,1,1);
plot(real(X));grid on;
subplot(2,1,2);
plot(angle(X));grid on;