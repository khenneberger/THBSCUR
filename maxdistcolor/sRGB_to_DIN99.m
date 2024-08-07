function Lab99 = sRGB_to_DIN99(rgb,XYZ,Lab,space,wpt)
% Convert an array of sRGB or CIEXYZ or CIELab values into DIN99 values (DIN 6176).
%
% (c) 2018-2024 Stephen Cobeldick
%
%%% Syntax:
% Lab99 = sRGB_to_DIN99(rgb)
% Lab99 = sRGB_to_DIN99([],XYZ)
% Lab99 = sRGB_to_DIN99([],[],Lab)
% Lab99 = sRGB_to_DIN99(..,..,..,space)
% Lab99 = sRGB_to_DIN99(..,..,..,..,wpt)
%
% <https://de.wikipedia.org/wiki/DIN99-Farbraum>
%
%% Inputs and Outputs
%
%%% Input Arguments (**=default):
% rgb = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes sRGB values R G B in the range 0<=RGB<=1.
% XYZ = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes CIEXYZ values X Y Z in the range 0<=Y<=1.
% Lab = Double/single array of size Nx3 or RxCx3, where the last
%       dimension encodes CIELAB values L* a* b* in the range 0<=L*<=100.
% space = 'DIN99'**/'DIN99o'/'DIN99b'/'DIN99c'/'DIN99d'/'ASTMD2244-07',
%         the name of a uniform colorspace based on DIN99.
% wpt = Double/single vector of the reference whitepoint, may be
%       supplied either as [x,y] or as [X,Y,Z] values. **D65 
%
%%% Output Argument:
% Lab99 = Array of same class and size as <rgb>/<XYZ>/<Lab>, where the last
%         dimension encodes the DIN99 values L99 a99 b99.
%
% See also DIN99_TO_SRGB SRGB_TO_CIELAB SRGB_TO_OKLAB SRGB_TO_OSAUCS
% SRGB_TO_CAM02UCS MAXDISTCOLOR MAXDISTCOLOR_VIEW MAXDISTCOLOR_DEMO

%% Input Wrangling %%
%
if nargin<5
	wpt = [];
end
%
if nargin<4 || isequal(space,'')
	space = 'DIN99';
end
switch upper(space)
	case 'ASTMD2244-07'
		c0=NaN;  c1=105.509; c2=0.0158; c3=16; c4=0.70; c5= 1;   c6=0.045; c7= 0; c8=0.045;
	case 'DIN99'
		c0=NaN;  c1=105.51;  c2=0.0158; c3=16; c4=0.70; c5= 1;   c6=0.045; c7= 0; c8=0.045;
	case {'O','DIN99O'}
		c0=NaN;  c1=303.67;  c2=0.0039; c3=26; c4=0.83; c5= 1;   c6=0.075; c7=26; c8=0.0435;
	case {'B','DIN99B'}
		c0=NaN;  c1=303.67;  c2=0.0039; c3=26; c4=0.83; c5=23;   c6=0.075; c7=26; c8=1;
	case {'C','DIN99C'}
		c0=0.1;  c1=317.65;  c2=0.0037; c3= 0; c4=0.94; c5=23;   c6=0.066; c7= 0; c8=1;
	case {'D','DIN99D'}
		c0=0.12; c1=325.22;  c2=0.0036; c3=50; c4=1.14; c5=22.5; c6=0.060; c7=50; c8=1;
	otherwise
		error('SC:sRGB_to_DIN99:space:NotRecognised',...
			'The 4th input "%s" is not a recognised DIN99 space.',space)
end
% Source: <https://www.researchgate.net/publication/229891006_Uniform_colour_spaces_based_on_the_DIN99_colour-difference_formula
%
if isequal(rgb,[])
	if isequal(XYZ,[])
		assert(isnan(c0),...
			'SC:sRGB_to_DIN99:Lab:Not_DIN99c_DIN99d',...
			'3rd input <Lab> is not supported for DIN99c and DIN99d.')
		assert(isfloat(Lab)&&isreal(Lab),...
			'SC:sRGB_to_DIN99:Lab:NotRealFloat',...
			'3rd input <Lab> must be a real floating-point array.')
		isz = size(Lab);
		assert(isz(end)==3||isequal(isz,[3,1]),...
			'SC:sRGB_to_DIN99:Lab:InvalidSize',...
			'3rd input <Lab> must have size Nx3 or RxCx3 or 3x1.')
		Lab = reshape(Lab,[],3);
		assert(all(Lab(:,1)>=0&Lab(:,1)<=100),...
			'SC:sRGB_to_DIN99:Lab:OutOfRange_L',...
			'3rd input <Lab> L values must be within the range 0<=L<=100')
	else
		assert(nargin<3||isequal(Lab,[]),...
			'SC:sRGB_to_DIN99:TooManyMatrices_Lab',...
			'If supplying <XYZ> values then <Lab> must be [].')
		if isfinite(c0)
			assert(isfloat(XYZ)&&isreal(XYZ),...
				'SC:sRGB_to_DIN99:XYZ:NotRealFloat',...
				'2nd input <XYZ> must be a real floating-point array.')
			isz = size(XYZ);
			assert(isz(end)==3||isequal(isz,[3,1]),...
				'SC:sRGB_to_DIN99:XYZ:InvalidSize',...
				'2nd input <XYZ> must have size Nx3 or RxCx3 or 3x1.')
			XYZ = reshape(XYZ,[],3);
			XYZ(:,1) = (1+c0).*XYZ(:,1) - c0.*XYZ(:,3);
		end
		Lab = sRGB_to_CIELab([],XYZ,wpt); % this does input checking :)
		Lab = reshape(Lab,[],3);
		isz = size(XYZ);
	end
else % RGB2XYZ
	assert(nargin<2||isequal(XYZ,[]),...
		'SC:sRGB_to_DIN99:TooManyMatrices_XYZ',...
		'If supplying <rgb> values then <XYZ> must be [].')
	assert(nargin<3||isequal(Lab,[]),...
		'SC:sRGB_to_DIN99:TooManyMatrices_Lab',...
		'If supplying <rgb> values then <Lab> must be [].')
	if isfinite(c0)
		XYZ = sRGB_to_CIEXYZ(rgb); % this does input checking :)
		XYZ = reshape(XYZ,[],3);
		XYZ(:,1) = (1+c0).*XYZ(:,1) - c0.*XYZ(:,3);
		Lab = sRGB_to_CIELab([],XYZ,wpt); % this does input checking :)
	else
		Lab = sRGB_to_CIELab(rgb,[],wpt); % this does input checking :)
	end
	Lab = reshape(Lab,[],3);
	isz = size(rgb);
end
%
% Lab2DIN99
kCH = 1;
kE  = 1;
L99 = c1.*log(1+c2*Lab(:,1))./kE;
e =     (Lab(:,2).*cosd(c3)+Lab(:,3).*sind(c3));
f = c4.*(Lab(:,3).*cosd(c3)-Lab(:,2).*sind(c3));
G = hypot(e,f);
C99 = c5.*log(1+c6*G)./(c8*kCH*kE);
h99 = atan2deg(f,e)+c7;
a99 = C99.*cosd(h99);
b99 = C99.*sind(h99);
% Source: <https://de.wikipedia.org/wiki/DIN99-Farbraum>
%
Lab99 = reshape([L99,a99,b99],isz);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sRGB_to_DIN99
function ang = atan2deg(Y,X)
% ATAN2 with an output in degrees. Note: ATAN2D only introduced R2012b.
try
	ang = atan2d(Y,X);
catch
	ang = atan2(Y,X)*180/pi;
	ang(Y==0 & X==0) = 0;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%atan2deg