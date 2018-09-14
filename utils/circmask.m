function mask = circmask(imgsize, cx, cy, r)

nx = imgsize(1);
ny = imgsize(2);
[X, Y] = meshgrid(0:nx-1, 0:ny-1);%-nx/2+1 : nx/2, -ny/2+1:ny/2);
R = sqrt((X+1-cx).^2 + (Y+1-cy).^2);
mask = double( R <= r );