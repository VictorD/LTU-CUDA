inputImage = single(imread('test.pgm'));

CUTIME = 1:99;
MATIME = 1:99;

for i=1
tic;
lcuda = cudaimalloc(inputImage)
cudaimerode(lcuda, uint8(43));
cudres = cudaimget(lcuda);
cudaimfree(lcuda);
CUTIME(i) = toc;
cudres = cudres(513:1536,513:2560);


tic;
maskSize = sqrt(2)*(2*i+1);
maskSize = ceil((maskSize + 1)/2)*2 - 3
matres = imerode(inputImage, strel('line', 61, -45));
MATIME(i) = toc;

isequal(matres,cudres)

end
clear cudaimalloc
clear cudaimerode
clear cudaimfree
clear cudaimget