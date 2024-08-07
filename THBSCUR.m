function [M, band_set, iter] =THBSCUR(Y,opts)

% Fast Hyperspectral Band Selection based on matrix CUR decomposition

% --------------------------------------------
% Input:
%                  Y       -    HSI band matrix
%           
%           opts           -   Structure value in Matlab. The fields are
%           opts.tol       -   termination tolerance
%           opts.max_iter  -   maximum number of iterations
%           opts.beta      -   stepsize for dual variable updating in ADMM
%           opts.lambda    -   lambda for sparse component 
%           opts.gamma1    -   parameter for spectral GL 
%           opts.gamma2    -   parameter for spatial GL 
%           opts.tau       -   step size in CUR gradient descent 
%           opts.k         -   number of desired bands 
%
% Output:
%               M          -   desired band set M=Y(:,k)
%               band_set   -   set of unique band indicies
%               iter       -   number of iterations run


%% Read Parameters
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'beta');        beta = opts.beta;            end
if isfield(opts, 'tau');         tau = opts.tau;              end
if isfield(opts, 'k');           k = opts.k;                  end
if isfield(opts, 'p');           p = opts.p;                  end
if isfield(opts, 'lambda1');      lambda1 = opts.lambda1;        end
if isfield(opts, 'lambda2');      lambda2 = opts.lambda2;        end





%% Initialize
dim = size(Y);
[d1,d2,d3] = size(Y);
rs = min(d1,round(k*log(d1)));
cs = min(d2,round(k*log(d2)));
B = zeros(dim);
S = B;
X = zeros(d1,d2,d3,3);
Xhat = X;
for i =1:3
    X(:,:,:,i) = B;
    Xhat(:,:,:,i) = B;
end

%% Random sampling for CUR
rng(1);
I = randperm(d1,rs); 
J = randperm(d2,cs);
C = Y(:,J,:); 
R = Y(I,:,:);
U = 0.5*(C(I,:,:)+R(:,J,:)); 
pinvU = ten_pinv(U);
B = htprod_fft(C,htprod_fft(pinvU,R));


%% ADMM
for iter = 1 : max_iter
  
   
    Bk = B;
        
    %% update B (low rank component)
    tmp = B-Y+S;
    grad_f = tmp;
        for i = 1:3
            grad_component = nabla_i(B, i) - X(:,:,:,i) + Xhat(:,:,:,i);% Compute the gradient along dimension i
            grad_f = grad_f + beta * nabla_i_transpose(grad_component, i);
        end
    GI = grad_f(I,:,:);
    GJ = grad_f(:,J,:);
    
    C = C-tau*GJ;
    R = R-tau*GI;
    U = 0.5*(C(I,:,:)+R(:,J,:)); 
    pinvU = ten_pinv(U);
    B = htprod_fft(C,htprod_fft(pinvU,R));
    if sum(isnan(B(:))) > 0
        B = Bk;
        break;
    end
    %% update S
    S = prox_l1(Y-B,lambda1/beta);
    
    %% update X,Xhat
    for i = 1:3
        gradi_B = nabla_i(B, i);
        X(:,:,:,i)= prox_pthv5(gradi_B+Xhat(:,:,:,i),lambda2/beta,p);
        Xhat(:,:,:,i)=Xhat(:,:,:,i)+X(:,:,:,i)-gradi_B;
    end
    %% Check for convergence
    
    chg = max(abs(Bk(:)-B(:)));
    
    if chg < tol
        break;
    end 
    
    %% update Zhat
    for i = 1:3
        gradi_B = nabla_i(B, i);
        Xhat(:,:,:,i)=Xhat(:,:,:,i)+X(:,:,:,i)-gradi_B;
    end
end
% convert to matrix to do k-means
[n1,n2,n3] = size(B);
n = n1*n2;
B2= reshape(B,n,n3);

%% Compute k-means clustering
[~, ~, ~, D] = kmeans(B2.', k, 'MaxIter', 100, 'Replicates', 50, 'EmptyAction', 'singleton');
[~,I] = min(D); % find indices for bands with centroids
% Ensure unique bands are selected
unique_band_indices = unique(I);  % Get unique indices
num_unique_bands = length(unique_band_indices);

if num_unique_bands < k
    % If not enough unique bands, select additional bands based on the next smallest distances in D
    disp('not unique/n')
    remaining_indices = setdiff(1:size(D, 1), unique_band_indices);
    [~, sorted_idx] = sort(D(remaining_indices, :), 'ascend');
    needed_indices = sorted_idx(1:(k - num_unique_bands));
    band_set = [unique_band_indices, remaining_indices(needed_indices)];
else
    band_set = unique_band_indices(1:k);  % Select only k bands if there are more than k unique bands
end

M = Y(:, band_set);  % Select bands from original data
%% 
    function grad = nabla_i(B, i)
    % Computes the gradient of B along the i-th dimension using diff, with boundary conditions
    switch i
        case 1  % Gradient along the first dimension
            grad = diff(B, 1, 1);  % Standard difference operation
            grad = cat(1, grad(1,:,:), grad);  % Add first row to the top, simulating forward difference at the boundary
        case 2  % Gradient along the second dimension
            grad = diff(B, 1, 2);  % Standard difference operation
            grad = cat(2, grad(:,1,:), grad);  % Add first column to the front, simulating forward difference at the boundary
        case 3  % Gradient along the third dimension
            grad = diff(B, 1, 3);  % Standard difference operation
            grad = cat(3, grad(:,:,1), grad);  % Add first layer to the front, simulating forward difference at the boundary
        otherwise
            error('Invalid dimension');
    end
end

    function result = nabla_i_transpose(grad, i)
    % Computes the transpose of the i-th gradient operator, ensuring the result is the same size as grad
    sizeB = size(grad);
    result = zeros(sizeB);  % Initialize result with the same size as grad/B

    switch i
        case 1  % Transpose gradient along the first dimension
            result(1,:) = grad(1,:);  % First boundary condition
            result(2:end-1,:) = grad(2:end-1,:) - grad(1:end-2,:);  % Central part
            result(end,:) = -grad(end-1,:);  % Last boundary condition
        case 2  % Transpose gradient along the second dimension
            result(:,1) = grad(:,1);  % First boundary condition
            result(:,2:end-1) = grad(:,2:end-1) - grad(:,1:end-2);  % Central part
            result(:,end) = -grad(:,end-1);  % Last boundary condition
        case 3  % Transpose gradient along the third dimension
            result(:,:,1) = grad(:,:,1);  % First boundary condition
            result(:,:,2:end-1) = grad(:,:,2:end-1) - grad(:,:,1:end-2);  % Central part
            result(:,:,end) = -grad(:,:,end-1);  % Last boundary condition
        otherwise
            error('Invalid dimension');
    end
end
end