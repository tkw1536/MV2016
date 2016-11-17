close all;
clear all;

A_c= imread('turing.png');
A = rgb2gray(A_c);
A = double(A)./255; % image
figure(1);
visualize_spectrum(A);


% h = fspecial('gaussian', hsize, sigma)
% h = fspecial('sobel') 
% h = fspecial('log', hsize, sigma) returns a rotationally symmetric Laplacian of Gaussian filter of size hsize with standard deviation sigma (positive). 
% hsize can be a vector specifying the number of rows and columns in h, or it can be a scalar, in which case h is a square matrix. The default value for hsize is [5 5] and 0.5 for sigma.
% size should be odd
%
sigma_spatial= 5;
B = fspecial('gaussian', [7*sigma_spatial 7*sigma_spatial], sigma_spatial); % some 2D filter function
figure(2);
visualize_spectrum(B);


[m,n] = size(A);
[mb,nb] = size(B); 

% output size: the full size of convolution  
mm = m + mb - 1;
nn = n + nb - 1;

% Note: Y = fft2(X,m,n) truncates X, or pads X with zeros to create an m-by-n array before doing the transform. The result is m-by-n.
Af= fft2(A,mm,nn);
Bf= fft2(B,mm,nn);

C = ifft2(Af.* Bf);
figure(3)
imshow(C,[]); % Display the result
title('Convolution done in Frequency Domain (Padded)');

% padding constants (for output of size == size(A))
% This is the top-left offset in C
padC_m = ceil((mb-1)./2);
padC_n = ceil((nb-1)./2);

% frequency-domain convolution result
D = C(padC_m+1:m+padC_m, padC_n+1:n+padC_n); 
figure(4); % imshow(D,[]);
title('Convolution done in Frequency Domain (Clipped Padding)');
visualize_spectrum(D);



%Now, compare the above with doing spatial-domain convolution, using conv2D

% % space-domain convolution result
F = conv2(A,B,'same');
% % same: Returns the central part of the convolution of the same size as A.
% % valid: Returns only those parts of the convolution that are computed without the zero-padded edges. Using this option, C has size [ma-mb+1,na-nb+1] when all(size(A) >= size(B)). Otherwise conv2 returns [].
% %        ma-mb+1= ma - 2(mb-1)/2
figure(5); imshow(F,[]);
title('Convolution done in Spacial Domain');

error= norm(F - D)





