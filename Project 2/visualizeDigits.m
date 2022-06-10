load digits.mat
i = 1; %select the digit to display
I = digits( i, : );
imagesc( reshape( I, 20, 20 ) );
colormap( gray );
axis image;

