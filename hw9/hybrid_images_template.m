% This is the solution of the lab exercise.
close all;
clear all;

%% L_ stands for the image on which a low-pass filter is applied.
L_c= imread('marilyn.bmp'); % 'dog.jpg'
is_color= size(size(L_c),2) == 3
L= L_c;
if (is_color)
    L = rgb2gray(L_c);
end;
L = double(L)./255; % image is now of floats in the range 0 to 1.
figure(2);
visualize_spectrum(L);

%% H_ stands for the image on which a high-pass filter is applied.
H_c= imread('einstein.bmp'); % 'cat.jpg'
is_color= size(size(H_c),2) == 3
H= H_c;
if (is_color)
    H = rgb2gray(H_c);
end;
H = double(H)./255; % image
figure(1); 
visualize_spectrum(H);


[m,n] = size(L);

%% This is the low-pass Gaussian filter in frequency domain.
sigma_l= 12; % Fill-in: The standard-deviation of the low-pass filter. Range 5-20.
figure(3); subplot(1,2,1);
LowPass_f = fspecial('gaussian', [m n], sigma_l); 
imshow(mat2gray(LowPass_f));  subplot(1,2,2);
LowPass_f = fftshift(LowPass_f); 
imshow(mat2gray(LowPass_f));

%% Pad, multiply and transform back to create a blurry image.
L_f= fft2(L,m,n);
Low_L_f= L_f.* LowPass_f;
C = ifft2(Low_L_f); 
figure(4); 
visualize_spectrum(real(C));
 
%% This is the high-pass Gaussian filter in frequency domain.
% Create an inverted Gaussian surface and normalize it 
% so that the sum of its elements in 1.
sigma_h= 12; % Fill-in: The standard-deviation of the high-pass filter. Range 5-20.
HighPass = fspecial('gaussian', [m n], sigma_h); 

% Invert and normalise HighPass_f
HighPass_f= max(max(HighPass)) - HighPass; 
HighPass_f = HighPass_f / sum(sum(HighPass_f)); 
figure(5);
imshow(mat2gray(log(1+abs(HighPass_f)))); 
% Take the frquency origin back to the top-left by applying fftshift
HighPass_f = fftshift(HighPass_f); 

%% Apply the high-pass Gaussian filter to the (Einstein) image.
H_f= fft2(H,m,n);
High_H_f= H_f.* HighPass_f;

% visualize your High_H_f
Ch= ifft2(High_H_f);
figure(6);
visualize_spectrum(real(Ch));

%% Now linearly combine the filtered frequency domain images
% Low_L_f and High_H_f to create the energy domain 
energy_l= norm(Low_L_f, 'fro')
energy_h= norm(High_H_f, 'fro')
weight=  0.05 % Fill-in: find a suitable weight which makes the result look good. Range (0,0.2]

% Hybrid image in frequency domain:
Hybrid_f= Low_L_f + High_H_f .*(weight*(energy_l/energy_h)); 
% Haybrid image in spatial domain:
Hybrid= real(ifft2(Hybrid_f));

figure(7); 
imshow(mat2gray(real(Hybrid)));
