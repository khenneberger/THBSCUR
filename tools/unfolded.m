function unfA = unfolded(A)
% unfold A along last dim

m = size(A);
lengthdim = length(m);
lastdim = m(lengthdim);
if lengthdim ==4
unfA = cat(1, A(:,:,:,1), A(:,:,:,2));

if lastdim >=3
    for i = 3:lastdim
        unfA = cat(1, unfA, A(:,:,:,i));
    end
end
end

if lengthdim ==3
unfA = cat(1, A(:,:,1), A(:,:,2));

if lastdim >=3
    for i = 3:lastdim
        unfA = cat(1, unfA, A(:,:,i));
    end
end
end

end