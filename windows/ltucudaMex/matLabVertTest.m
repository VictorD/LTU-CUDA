inputImage = single(imread('square.pgm'));
inputImage = inputImage(1:2048, 1:2048);

CUTIME_D = 1:99;
MATIME_D = 1:99;

for i=37
cudaimfree(); % reset device to be sure

mstrel = strel('line', 2*i+1, 45); %strel('line', 2*i+1, -45);
lcuda = cudaimalloc();
tic;
cudaimcopy(lcuda, inputImage);
cudaimerode(lcuda, cudastrel(mstrel));
cudres = cudaimget(lcuda);
CUTIME_D(i-2) = toc;
cudaimfree(lcuda);

tic;
matres = imerode(inputImage, mstrel);
MATIME_D(i-2) = toc;

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