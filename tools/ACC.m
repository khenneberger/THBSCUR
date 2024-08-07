function c = ACC(M)

% Average Correlation Coeficient for Hyperspectral Band Selection
% (Quantitative Metric)
% Input:    M - matrix of selected bands
% Outut:    c - Average Correlation Coeficient (qualtifies the intraband
% correlation)

C = corrcoef(M);
c = mean2(C);