function Lab = sRGB_to_OKLab(rgb,XYZ)
% Convert an array of sRGB or CIEXYZ values into OKLab values.
%
% (c) 2018-2024 Stephen Cobeldick
%
%%% Syntax:
% Lab = sRGB_to_OKLab(rgb)
% Lab = sRGB_to_OKLab([],XYZ)
%
% <https://bottosson.github.io/posts/oklab/>
%
%% Inputs and Outputs
%
%%% Input Argument:
% rgb = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes sRGB values R G B in the range 0<=RGB<=1.
% XYZ = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes CIEXYZ values X Y Z in the range 0<=Y<=1.
%
%%% Output Argument:
% Lab = Array of same class and size as <rgb>, where the last dimension
%       encodes OKLAB values [L,a,b] in the range 0<=L<=1.
%
% See also OKLAB_TO_SRGB SRGB_TO_CIELAB SRGB_TO_DIN99 SRGB_TO_OSAUCS
% SRGB_TO_CAM02UCS MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
if isequal(rgb,[])
	assert(isfloat(XYZ)&&isreal(XYZ),...
		'SC:sRGB_to_OKLab:XYZ:NotRealFloat',...
		'2nd input <XYZ> must be a real floating-point array.')
	isz = size(XYZ);
	assert(isz(end)==3||isequal(isz,[3,1]),...
		'SC:sRGB_to_OKLab:XYZ:InvalidSize',...
		'2nd input <XYZ> must have size Nx3 or RxCx3 or 3x1.')
	XYZ = reshape(XYZ,[],3);
	assert(all(XYZ(:,2)>=0&XYZ(:,2)<=1),...
		'SC:sRGB_to_OKLab:XYZ:OutOfRange_Y',...
		'The XYZ Y values must be within the range 0<=Y<=1.')
else % RGB2XYZ
	assert(nargin<2||isequal(XYZ,[]),...
		'SC:sRGB_to_OKLab:InvalidInputs',...
		'If supplying <rgb> then <XYZ> must be [].')
	XYZ = sRGB_to_CIEXYZ(rgb); % this does input checking :)
	isz = size(rgb);
	XYZ = reshape(XYZ,[],3);
end
%
% XYZ2OKLab
M1 = [... XYZ to approximate cone responses:
	+0.8189330101, +0.3618667424, -0.1288597137;...
	+0.0329845436, +0.9293118715, +0.0361456387;...
	+0.0482003018, +0.2643662691, +0.6338517070];
M2 = [... nonlinear cone responses to Lab:
	+0.2104542553, +0.7936177850, -0.0040720468;...
	+1.9779984951, -2.4285922050, +0.4505937099;...
	+0.0259040371, +0.7827717662, -0.8086757660];
lms = XYZ * M1.';
lmsp = nthroot(lms,3);
Lab = lmsp * M2.';
Lab = reshape(Lab,isz);
% source: <https://bottosson.github.io/posts/oklab/>
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_OKLab