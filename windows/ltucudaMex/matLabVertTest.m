inputImage = single(imread('square.pgm'));

CUTIME_D = 1:99;
MATIME_D = 1:99;

for i=3:101
cudaimfree(); % reset device to be sure

matstrel = strel('line', 2*i+1, -45);
[width height] = size(matstrel.getnhood);

tic;
lcuda = cudaimalloc(inputImage);
cudaimerode(lcuda, uint8(width));
cudres = cudaimget(lcuda);
cudaimfree(lcuda);
CUTIME_D(i-2) = toc;

tic;
matres = imerode(inputImage, matstrel);
MATIME_D(i-2) = toc;

isequal(matres,cudres)

end
clear cudaimalloc
clear cudaimerode
clear cudaimfree
clear cudaimget

%exit;