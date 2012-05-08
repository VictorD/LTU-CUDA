function out = cudastrel(se_in)
se_elements = getsequence(se_in);
out = struct();
strelsizes = [];
se = [];
he = [];
num = 0;
for i=1:length(se_elements)
    nh = getnhood(se_elements(i));
    s = size(nh);
    strelsizes = cat(2, strelsizes, s);
    rs = reshape(nh, 1, s(1)*s(2));
    se = cat(2, se, rs);

    % Section added for non-flat SE support.
    sh = getheight(se_elements(i));
    rsh = reshape(sh, 1, s(1)*s(2));
    he = cat(2, he, rsh);
    
    num = num + 1;
end

out.data = uint8(se);
out.sizes = int32(strelsizes);
out.heights = single(he);
out.num = int32(num);
out.isFlat = int32(se_elements.isflat);


o = out.heights;
