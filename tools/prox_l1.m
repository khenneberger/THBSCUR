function x = prox_l1(b,lambda)

% The proximal operator of the l1 norm
% 
% min_x lambda*||x||_1+0.5*||x-b||_2^2
% 
x = sign(b).*max(abs(b)-lambda,0);