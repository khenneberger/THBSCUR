function ten_A_inv = ten_pinv(A)
[d1,d2,d3] = size(A);
A = fft(A,[],3);
A_inv = zeros(d2,d1,d3);
for i = 1:d3
    A_inv(:,:,i) = pinv(A(:,:,i));
end
ten_A_inv = ifft(A_inv,[],3);
end