%%*****************************************
%% Lucas and Kanade's Optical flow in Rush Hour.
%% K. Pathak: This is a heavily modified version of codes from:
%% 1. http://www.cs.ucf.edu/~mikel/Research/Optical_Flow.htm
%% 2. https://de.mathworks.com/matlabcentral/fileexchange/48745-lucas-kanade-tutorial-example-2/
%% The "Rush Hour" video is from: https://www.youtube.com/watch?v=ufK2XRGUjuc
%%*****************************************

clear all;

vidObj = VideoReader('videos/cars_at_an_intersection_hirez.mp4');
vidHeight = vidObj.Height
vidWidth = vidObj.Width

vidWriter = VideoWriter('cars_at_an_intersection_optical_flow.avi');
vidWriter.FrameRate = 0.75*vidObj.FrameRate;
open(vidWriter);

vidObj.CurrentTime = 1.0;

while hasFrame(vidObj)
    
    ORIGINAL_IMAGE_COL_1= readFrame(vidObj);
    if vidObj.CurrentTime + 0.05 <= vidObj.Duration
        vidObj.CurrentTime = vidObj.CurrentTime + 0.05;
        ORIGINAL_IMAGE_COL_2= readFrame(vidObj);
    else
        break;
    end;
    
    %%Image Variables:
    
    ORIGINAL_IMAGE_1=im2double(rgb2gray(ORIGINAL_IMAGE_COL_1));
    ORIGINAL_IMAGE_2=im2double(rgb2gray(ORIGINAL_IMAGE_COL_2));
    
    [height,width]=size(ORIGINAL_IMAGE_1);
    
    gauss_sigma = 1;
    IMAGE_1_SMOOTHED=zeros(height,width);
    IMAGE_2_SMOOTHED=zeros(height,width);
    
    %%Derivate Variables:
    Dx_1=zeros(height,width);
    Dy_1=zeros(height,width);
    Dx_2=zeros(height,width);
    Dy_2=zeros(height,width);
    
    Ix=zeros(height,width);
    Iy=zeros(height,width);
    It=zeros(height,width);
    
    
    %%Optical flow variables
    neighborhood_size=5;
    
    A=zeros(2,2);
    B=zeros(2,1);
    
    %%Kernel Variables:
    Kernel_Size = 6*gauss_sigma+1;
    k = (Kernel_Size-1)/2;
    gauss_kernel_x=zeros(Kernel_Size,Kernel_Size);
    gauss_kernel_y=zeros(Kernel_Size,Kernel_Size);
    kernel=zeros(Kernel_Size,Kernel_Size);
    
    % Make a kernel for partial derivatve
    % Of Gaussian with respect to x (for computing Dx)
    for i=1:Kernel_Size
        for j=1:Kernel_Size
            gauss_kernel_x(i,j) = -( (j-k-1)/( 2* pi * gauss_sigma^3 ) ) * exp ( - ( (i-k-1)^2 + (j-k-1)^2 )/ (2*gauss_sigma^2) );
        end
    end
    
    % Make a kernel for partial derivatve
    % Of Gaussian with respect to y (for computing Dy)
    for i=1:Kernel_Size
        for j=1:Kernel_Size
            gauss_kernel_y(i,j) = -( (i-k-1)/( 2* pi * gauss_sigma^3 ) ) *  exp ( - ( (i-k-1)^2 + (j-k-1)^2 )/ (2*gauss_sigma^2) );
        end
    end
      
    %%Compute x and y derivates for both images:  
    Dx_1 = conv2(gauss_kernel_x,ORIGINAL_IMAGE_1); % not filter2
    Dy_1 = conv2(gauss_kernel_y,ORIGINAL_IMAGE_1);
    Dx_2 = conv2(gauss_kernel_x,ORIGINAL_IMAGE_2);
    Dy_2 = conv2(gauss_kernel_y,ORIGINAL_IMAGE_2);
    
    
    Ix = (Dx_1 + Dx_2) / 2;
    Iy = (Dy_1 + Dy_2) / 2;
    
    
    %figure(2);
    %subplot(2,1,1);
    %imshow(Ix, []);
    %title('I_x');
    %subplot(2,1,2);
    %imshow(Iy, []);
    %title('I_y');
    
    %% Build a Gaussian kernel to smooth images for computing It
    for i=1:Kernel_Size
        for j=1:Kernel_Size
            kernel(i,j) = (1/(2*pi*(gauss_sigma^2))) * exp (-((i-k-1)^2 + (j-k-1)^2)/(2*gauss_sigma^2));
        end
    end
    
    IMAGE_1_SMOOTHED = conv2(kernel,ORIGINAL_IMAGE_1);
    IMAGE_2_SMOOTHED = conv2(kernel,ORIGINAL_IMAGE_2);
    
    C1 = corner(ORIGINAL_IMAGE_1, 'MinimumEigenvalue'); 
    % 'MinimumEigenvalue': Shi-Tomasi corners.
    % 'Harris': Harris Corners. 
    % Optical-flow is only found at these corners 
    
    %figure(1);
    %subplot(2,1,1);
    %imshow(IMAGE_1_SMOOTHED, []);   
    %hold on;
    %plot(C1(:,1), C1(:,2), 'bx');    
    %subplot(2,1,2);
    %imshow(IMAGE_2_SMOOTHED, []);
    
    % I_t the partial derivative w.r.t. time.
    It = IMAGE_2_SMOOTHED - IMAGE_1_SMOOTHED;
    %figure(3);
    %imshow(It, [min(It(:)), max(It(:))]);
    %title('I_t Difference Image');
    
    v_x= zeros(size(C1,1),1);
    v_y= zeros(size(C1,1),1);
    
    for c_ix= 1: size(C1, 1)
        i= C1(c_ix, 2); % y-coord of the corner is the row
        j= C1(c_ix, 1); % x-coord of the corner is the col
        A=zeros(2,2);
        B=zeros(2,1);
        
        for m=i-floor(neighborhood_size/2):i+floor(neighborhood_size/2)
            for n=j-floor(neighborhood_size/2):j+floor(neighborhood_size/2)
                if (m < 1) | (m > height) | (n < 1) | (n > width)
                    continue;
                end
                A(1,1)=A(1,1) + Ix(m,n)*Ix(m,n);
                A(1,2)=A(1,2) + Ix(m,n)*Iy(m,n);
                A(2,1)=A(2,1) + Ix(m,n)*Iy(m,n);
                A(2,2)=A(2,2) + Iy(m,n)*Iy(m,n);
                
                B(1,1)=B(1,1) + Ix(m,n)*It(m,n);
                B(2,1)=B(2,1) + Iy(m,n)*It(m,n);
                
            end
        end
        
        Ainv= pinv(A); %%Pseudo inverse
        result= Ainv*(-B);
        v_x(c_ix)= result(1,1);
        v_y(c_ix)= result(2,1);
    end
    
    %figure(4);
    imshow(ORIGINAL_IMAGE_COL_1);
    hold on;
    quiver(C1(:,1), C1(:,2), v_x, v_y, 1,'r');
    img_frame= getframe;
    writeVideo(vidWriter, img_frame);
    
    pause(1/vidObj.FrameRate);

end

close(vidWriter);