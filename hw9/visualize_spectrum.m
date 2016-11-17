function visualize_spectrum(B)

subplot(1,2,1)
imshow(B, [])
title ('Image (x,y)');

Bf = fft2(B);
Bf = fftshift(Bf); % Center FFT for visualization

Bf = abs(Bf); % Get the magnitude
Bf = log(Bf+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined

Bf = mat2gray(Bf); % Use mat2gray to scale the image between 0 and 1
subplot(1,2,2)
imshow(Bf, []); % Display the result

%ax= subplot(1,2,2);
%surf(Bf); colormap(ax, jet); 
%view(2);
%daspect([1 1 0.01])
%colorbar
title ('Spectrum log(|F(u,v)| + 1)');
