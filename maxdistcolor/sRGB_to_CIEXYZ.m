function XYZ = sRGB_to_CIEXYZ(rgb)
% Convert a matrix of sRGB values to CIEXYZ tristimulus values.
%
% (c) 2018-2024 Stephen Cobeldick
%
%%% Syntax:
% XYZ = sRGB_to_CIEXYZ(rgb)
%
% <https://en.wikipedia.org/wiki/SRGB>
% <https://en.wikipedia.org/wiki/CIE_1931_color_space>
%
%% Inputs and Outputs
%
%%% Input Argument:
% rgb = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes sRGB values R G B  in the range 0<=RGB<=1.
%
%%% Output Argument:
% XYZ = Array of same class and size as <rgb>, where the last
%       dimension encodes CIEXYZ values X Y Z in the range 0<=Y<=1.
%
% See also CIEXYZ_TO_SRGB SRGB_TO_CIELAB SRGB_TO_DIN99 SRGB_TO_OSAUCS
% SRGB_TO_CAM02UCS MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
assert(isfloat(rgb)&&isreal(rgb),...
	'SC:sRGB_to_XYZ:rgb:NotRealFloat',...
	'1st input <rgb> array must be a real floating-point array.')
isz = size(rgb);
assert(isz(end)==3||isequal(isz,[3,1]),...
	'SC:sRGB_to_XYZ:rgb:InvalidSize',...
	'1st input <rgb> must have size Nx3 or RxCx3 or 3x1.')
rgb = reshape(rgb,[],3);
assert(all(rgb(:)>=0&rgb(:)<=1),...
	'SC:sRGB_to_XYZ:rgb:OutOfRange',...
	'1st input <rgb> values must be within the range 0<=rgb<=1')
%
%% RGB2XYZ %%
%
M = [... IEC 61966-2-1:1999
	0.4124,0.3576,0.1805;...
	0.2126,0.7152,0.0722;...
	0.0193,0.1192,0.9505];
% M = [... Derived from ITU-R BT.709-6
%    0.412390799265959,0.357584339383878,0.180480788401834;...
%    0.212639005871510,0.715168678767756,0.072192315360734;...
%    0.019330818715592,0.119194779794626,0.950532152249661];
% M = [... <http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html>
% 	0.4124564,0.3575761,0.1804375;...
% 	0.2126729,0.7151522,0.0721750;...
% 	0.0193339,0.1191920,0.9503041];
%
XYZ = sGammaInv(rgb) * M.';
XYZ = reshape(XYZ,isz);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_CIEXYZ
function out = sGammaInv(inp)
% Inverse gamma correction: Nx3 sRGB -> Nx3 linear RGB.
idx = inp > 0.04045;
out = inp / 12.92;
out(idx) = real(((inp(idx) + 0.055) ./ 1.055) .^ 2.4);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sGammaInv