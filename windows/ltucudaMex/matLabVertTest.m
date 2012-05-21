inputImage = single(imread('square.pgm'))/255.0;
inputImage = inputImage(1:1472, 1:1472);
inputImage = padarray(inputImage, [288 288], 1);

CUTIME = 1:99;
MATIME = 1:99;

for i=1:34
cudaimfree(); % reset device to be sure

matstrel = strel('octagon', 3*(2*i+1));

tmp = matstrel.getsequence;
lcuda = cudaimalloc();
tic;
cudaimcopy(lcuda, inputImage);
for j=1:length(tmp)
cudaimerode(lcuda, cudastrel(tmp(j)));
end
cudres = cudaimget(lcuda);
CUTIME(i) = toc;
cudaimfree(lcuda);

tic;
matres = imerode(inputImage, matstrel);
toc



MATIME(i) = toc;

isequal(matres,cudres)
if (isequal(matres,cudres) == 0)
    fprintf('Bad result\n');
end

end
clear cudaimalloc
clear cudaimcopy
clear cudaimerode
clear cudaimfree
clear cudaimget

%exit;