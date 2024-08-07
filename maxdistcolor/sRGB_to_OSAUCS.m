function Ljg = sRGB_to_OSAUCS(rgb,XYZ,isd,isc)
% Convert a matrix of sRGB or CIWXYZ values into OSA-UCS values.
%
% (c) 2020-2024 Stephen Cobeldick
%
%%% Syntax:
% Ljg = sRGB_to_OSAUCS(rgb)
% Ljg = sRGB_to_OSAUCS(rgb,[],isd)
% Ljg = sRGB_to_OSAUCS(rgb,[],isd,isc)
% Ljg = sRGB_to_OSAUCS([],XYZ,...)
%
% If the output is being used for calculating the Euclidean color distance
% (i.e. deltaE) then set isd=true, so that L is _not_ divided by sqrt(2).
%
% The reference formula divides by zero when Y0^(1/3)==2/3 (dark colors),
% this unfortunate numeric discontinuity can be avoided with isc=true.
%
% <https://en.wikipedia.org/wiki/OSA-UCS>
%
%% Inputs and Outputs
%
%%% Input Argument (**==default):
% rgb = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes sRGB values R G B in the range 0<=RGB<=1.
% XYZ = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes CIEXYZ values X Y Z in the range 0<=Y<=1.
% isd = LogicalScalar, true/false** = Euclidean distance/reference L values.
% isc = LogicalScalar, true/false** = modified continuous/reference values.
%
%%% Output Argument:
% Ljg = Array of same class and size as <rgb>/<XYZ>, where the last
%       dimension encodes OSA-UCS values L j g.
%
% See also OSAUCS_TO_SRGB SRGB_TO_CIELAB SRGB_TO_DIN99 SRGB_TO_OKLAB
% SRGB_TO_CAM02UCS MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
if isequal(rgb,[])
	assert(isfloat(XYZ)&&isreal(XYZ),...
		'SC:sRGB_to_OSAUCS:XYZ:NotRealFloat',...
		'2nd input <XYZ> must be a real floating-point array.')
	isz = size(XYZ);
	assert(isz(end)==3||isequal(isz,[3,1]),...
		'SC:sRGB_to_OSAUCS:XYZ:InvalidSize',...
		'2nd input <XYZ> must have size Nx3 or RxCx3 or 3x1.')
	XYZ = reshape(XYZ,[],3);
	assert(all(XYZ(:,2)>-0.1&XYZ(:,2)<+1.1),...
		'SC:sRGB_to_OSAUCS:XYZ:OutOfRange_Y',...
		'The <XYZ> Y values must be within the range 0<=Y<=1.')
else % RGB2XYZ
	assert(nargin<2||isequal(XYZ,[]),...
		'SC:sRGB_to_OSAUCS:InvalidInputs',...
		'If supplying <rgb> then <XYZ> must be [].')
	XYZ = sRGB_to_CIEXYZ(rgb); % this does input checking :)
	isz = size(rgb);
	XYZ = reshape(XYZ,[],3);
end
%
if nargin<3
	isd = false;
else
	assert(ismember(isd,0:1),...
		'SC:sRGB_to_OSAUCS:isd:NotScalarLogical',...
		'Third input <isd> must be true/false.')
end
%
if nargin<4
	isc = false;
else
	assert(ismember(isc,0:1),...
		'SC:sRGB_to_OSAUCS:isc:NotScalarLogical',...
		'Fourth input <isc> must be true/false.')
end
%
%% RGB2Ljg %%
%
% XYZ2Ljg
XYZ = 100*XYZ;
xyz = bsxfun(@rdivide,XYZ,sum(XYZ,2));
xyz(isnan(xyz)) = 0;
%
K = 1.8103 + (xyz(:,1:2).^2)*[4.4934;4.3034] - ...
	prod(xyz(:,1:2),2)*4.276 - xyz(:,1:2)*[1.3744;2.5643];
Y0 = K.*XYZ(:,2);
Lp = 5.9*(nthroot(Y0,3)-2/3 + 0.042*nthroot(Y0-30,3));
L = (Lp-14.3993)./sqrt(2-isd);
%C = 1 + (0.042*nthroot(Y0-30,3))./(nthroot(Y0,3)-2/3); % !!!!! divide by zero !!!!!
C = 1 + (0.042*nthroot(Y0-30,3))./(nthroot(max(30*isc,Y0),3)-2/3);
tmp = nthroot(XYZ*[0.799,0.4194,-0.1648;-0.4493,1.3265,0.0927;-0.1149,0.3394,0.717].',3);
a = tmp*[-13.7;17.7;-4];
b = tmp*[1.7;8;-9.7];
Ljg = reshape([L,C.*b,C.*a],isz);
% Reference: <https://en.wikipedia.org/wiki/OSA-UCS>
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_OSAUCS