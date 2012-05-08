inputImage = single(imread('square.pgm'));

CUTIME_V2 = 1:99;
MATIME_V = 1:99;

for i=71%3:101
cudaimfree(); % reset device to be sure

tic;
lcuda = cudaimalloc(inputImage);
cudaimerode(lcuda, uint8(2*i+1));
cudres = cudaimget(lcuda);
cudaimfree(lcuda);
CUTIME_V2(i-2) = toc;



tic;
matres = imerode(inputImage, strel('line', 2*i+1, 90));
MATIME_V(i-2) = toc;
isequal(matres,cudres)

end
clear cudaimalloc
clear cudaimerode
clear cudaimfree
clear cudaimget

%exit;