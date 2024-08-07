function u = prox_pthv5(x, lambda, p)

n = length(x);
if p==1
    u = sign(x).*max(abs(x)-lambda,0);
end
% switch p
%     case 2
%         u = sign(x).*max((abs(x)-(lambda/(1+(lambda*n*p)))*norm(x,1)),0);
%     case 3
%         u = sign(x).*((sqrt(1+(12*n*lambda*norm(x,1))))/(6*n*lambda));
% end
if p==2
    u = sign(x).*max((abs(x)-(2*lambda/(1+(lambda*n*p)))*sum(abs(x))),0);
end
if p==3
    u = sign(x).*max(abs(x)-3*lambda*((-1+sqrt(1+(12*n*lambda*sum(abs(x)))))/(6*n*lambda)).^2,0);
end
if p==4
    numerator = -2 * 3^(1/3) * 4*n*lambda + 2^(1/3) * (9 * (4*n*lambda)^2 * abs(x) + sqrt(3) * sqrt((4*n*lambda)^3 * (4 + 27 * 4*n*lambda * abs(x).^2))).^(2/3);
    denominator = 6^(2/3) * 4*n*lambda * (9 * (4*n*lambda)^2 * abs(x) + sqrt(3) * sqrt((4*n*lambda)^3 * (4 + 27 * 4*n*lambda * abs(x).^2))).^(1/3);
    result = numerator ./ denominator;
    u = sign(x).*max(abs(x)-3*lambda*(result).^2,0);
end