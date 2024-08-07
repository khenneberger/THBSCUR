function Lab = sRGB_to_CIELab(rgb,XYZ,wpt)
% Convert a matrix of sRGB or CIEXYZ values into CIELAB values.
%
% (c) 2018-2024 Stephen Cobeldick
%
%%% Syntax:
% Lab = sRGB_to_CIELab(rgb)
% Lab = sRGB_to_CIELab([],XYZ)
% Lab = sRGB_to_CIELab(..,..,wpt)
%
% <https://en.wikipedia.org/wiki/CIELAB_color_space>
%
%% Inputs and Outputs
%
%%% Input Arguments (**=default):
% rgb = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes sRGB values R G B in the range 0<=RGB<=1.
% XYZ = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes CIEXYZ values X Y Z in the range 0<=Y<=1.
% wpt = Double/single vector of the reference whitepoint, may be
%       supplied either as [x,y] or as [X,Y,Z] values. **D65 
%
%%% Output Argument:
% Lab = Array of same class and size as <rgb>/<XYZ>, where the last
%       dimension encodes CIELAB values L* a* b* in the range 0<=L*<=100.
%
% See also CIELAB_TO_SRGB SRGB_TO_DIN99 SRGB_TO_OKLAB SRGB_TO_OSAUCS
% SRGB_TO_CAM02UCS MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
if isequal(rgb,[])
	assert(isfloat(XYZ)&&isreal(XYZ),...
		'SC:sRGB_to_CIELab:XYZ:NotRealFloat',...
		'2nd input <XYZ> must be a real floating-point array.')
	isz = size(XYZ);
	assert(isz(end)==3||isequal(isz,[3,1]),...
		'SC:sRGB_to_CIELab:XYZ:InvalidSize',...
		'2nd input <XYZ> must have size Nx3 or RxCx3 or 3x1.')
else % RGB2XYZ
	assert(nargin<2||isequal(XYZ,[]),...
		'SC:sRGB_to_CIELab:InvalidInputs',...
		'If supplying <rgb> then <XYZ> must be [].')
	XYZ = sRGB_to_CIEXYZ(rgb); % this does input checking :)
	isz = size(rgb);
end
%
XYZ = reshape(XYZ,[],3);
assert(all(XYZ(:,2)>=0&XYZ(:,2)<=1),...
	'SC:sRGB_to_CIELab:XYZ:OutOfRange_Y',...
	'The <XYZ> Y values must be within the range 0<=Y<=1.')
%
if nargin<3 || isequal(wpt,[])
	wpt = [0.31272,0.32903]; % D65
else
	assert(isfloat(wpt)&&isreal(wpt),...
		'SC:sRGB_to_CIELab:wpt:NotRealFloat',...
		'3rd input <wpt> must be a real floating-point array.')
	assert(ismember(numel(wpt),2:3),...
		'SC:sRGB_to_CIELab:wpt:InvalidSize',...
		'3rd input <wpt> must be [x,y] or [X,Y,Z] whitepoint.')
	wpt = reshape(double(wpt),1,[]);
end
if numel(wpt)==2
	x = wpt(1);
	y = wpt(2);
	wpt = [x./y,1,(1-x-y)./y]; % xy to XYZ
else
	assert(wpt(2)==1,...
		'SC:sRGB_to_CIELab:wpt:OutOfRange_Y',...
		'3rd input <wpt> Y value must be exactly one.')
end
%
%% RGB2Lab
%
% XYZ2Lab
epsilon = 216/24389; % (6/29)^3
kappa   = 24389/27;  % (29/3)^3
% source: <http://www.brucelindbloom.com/index.html?LContinuity.html>
xyzr = bsxfun(@rdivide,XYZ,wpt);
idx  = xyzr>epsilon;
fxyz = (kappa*xyzr+16)/116;
fxyz(idx) = nthroot(xyzr(idx),3);
Lab = [116*fxyz(:,2)-16,...
	500*(fxyz(:,1)-fxyz(:,2)),...
	200*(fxyz(:,2)-fxyz(:,3))];
% source: <http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html>
%
Lab = reshape(Lab,isz);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_CIELab