function [Ls, Lt] = glapv2(D, pars)
%% create a pair of graph Laplacians in the spatiotemporal domain
% Inputs:
% D: video of size n1 x n2 x m with n1xn2 pixels and m frames
% pars.hs: spatial filtering parameter
%     .ht: temporal filtering parameter
%     .K:  number of nearest neighbors in the spatial domain

% Outputs:
% Ls: graph Laplacian in the spatial domain
% Lt: graph Laplacian in the temporal domain

%% Read parameters
Ks = pars.Ks; % k-nearest neighbor search
Kt = pars.Kt;
ht= pars.ht;
hs= pars.hs;

if size(D,3)>1
    [n1,n2,t] = size(D);
    n = n1*n2;

    D2 = reshape(D,n,t);
else
    D2 = D;
    [n,t] = size(D2);
end

%% Create Lt
[dt,ind] = pdist2(D2',D2','squaredeuclidean','Smallest',Kt+1);
tmp = exp(-dt(2:end,:)/ht^2);
idx = reshape(repmat((1:t),Kt,1),t*Kt,1); %row index
idy = reshape(ind(2:end,:),t*Kt,1); %col index
A = sparse(idx,idy,tmp(:),t,t);
A = A+A';
d = 1./sqrt(sum(A,2));
W = spdiags(d,0,t,t);
Lt = speye(t)-W*A*W;

%% Create Ls (pixel-based)
[ds, ind] = pdist2(D2,D2,'squaredeuclidean','Smallest',Ks+1);
tmp = exp(-ds(2:end,:)/hs^2);
idx = reshape(repmat((1:n),Ks,1),n*Ks,1); %row index
idy = reshape(ind(2:end,:),n*Ks,1);
A = sparse(idx,idy,tmp(:),n,n);
A = A+A';
d = 1./sqrt(sum(A,2));
W = spdiags(d,0,n,n);
Ls = speye(n)-W*A*W;

