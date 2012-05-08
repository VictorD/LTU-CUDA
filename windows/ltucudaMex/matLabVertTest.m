inputImage = single(imread('square.pgm'));

CUTIME_D = 1:99;
MATIME_D = 1:99;

for i=37
cudaimfree(); % reset device to be sure

mstrel = [strel('line', 5, 0); strel('line', 5, 90)] %strel('line', 2*i+1, -45);

tic;
lcuda = cudaimalloc(inputImage);
cudaimerode(lcuda, cudastrel(mstrel));
cudres = cudaimget(lcuda);
cudaimfree(lcuda);
CUTIME_D(i-2) = toc;

tic;
matres = imerode(inputImage, mstrel);
MATIME_D(i-2) = toc;

isequal(matres,cudres)

end
clear cudaimalloc
clear cudaimerode
clear cudaimfree
clear cudaimget

%exit;