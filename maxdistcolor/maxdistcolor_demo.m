%%% MAXDISTCOLOR Demo Script %%%
% Plot the colormap in the UCS, as a distance plot, and as colorbars with colornames.
%
% The CAM02UCS colorspace functions must be obtained separately, for
% example my implementation here: <https://github.com/DrosteEffect/CIECAM02>
%
%%% Define the colorspace function:
fnh = @sRGB_to_OKLab;   % OKLab  (recommended)
%fnh = @sRGB_to_CIELab; % CIELab (not very uniform colorspace)
%fnh = @(m)sRGB_to_CAM02UCS(m,true,'LCD');  % CAM02-LCD
%fnh = @(m)sRGB_to_CAM02UCS(m,true,'UCS');  % CAM02-UCS
%fnh = @(m)sRGB_to_OSAUCS(m,[],true,true);  % OSA-UCS (modified)
%fnh = @(m)sRGB_to_DIN99(m);                % DIN99
%fnh = @(m)sRGB_to_DIN99(m,[],[],'DIN99o'); % DIN99o
%
%%% Generate colormap of distinctive colors:
[rgb,ucs,sts] = maxdistcolor(9,fnh);
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'exc',[]);
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'class','single');
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'Cmin',0.5,'Cmax',0.6);
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'Lmin',0.4,'Lmax',0.6);
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'inc',[0,0,0;1,0,1],'exc',[0,1,0]);
%[rgb,ucs,sts] = maxdistcolor(9,fnh,'sort','longest','disp','verbose');
%[rgb,ucs,sts] = maxdistcolor(9,fnh, 'bitR',8,'bitG',8,'bitB',8); % Truecolor -> slow!
%[rgb,ucs,sts] = maxdistcolor(64,fnh,'bitR',2,'bitG',2,'bitB',2, 'exc',[]); % entire RGB gamut.
N = size(rgb,1);
%
%%% Plot color distance matrix:
figure();
for k = 1:N
	dst = sqrt(sum(bsxfun(@minus,ucs,ucs(k,:)).^2,2));
	scatter3(k*ones(1,N),1:N,dst, 123, rgb,...
		'MarkerFaceColor',rgb(k,:), 'LineWidth',2.8, 'Marker','o')
	hold on
end
title(sprintf('Colormap Euclidean Distances in %s Colorspace',sts.colorspace))
zlabel('Euclidean Distance')
ylabel('Colormap Index')
xlabel('Colormap Index')
set(gca,'XTick',1:N,'YTick',1:N)
%
%%% Plot colors in UCS:
figure();
scatter3(ucs(:,3),ucs(:,2),ucs(:,1), 256, rgb, 'filled')
text(ucs(:,3),ucs(:,2),ucs(:,1),cellstr(num2str((1:N).')), 'HorizontalAlignment','center')
%
%%% Plot outline of RGB cube:
M = 23;
[X,Y,Z] = ndgrid(linspace(0,1,M),0:1,0:1);
mat = fnh([X(:),Y(:),Z(:);Y(:),Z(:),X(:);Z(:),X(:),Y(:)]);
X = reshape(mat(:,3),M,[]);
Y = reshape(mat(:,2),M,[]);
Z = reshape(mat(:,1),M,[]);
line(X,Y,Z,'Color','k')
axis('equal')
title(sprintf('Colormap in %s Colorspace',sts.colorspace))
zlabel(sts.axesLabels{1})
ylabel(sts.axesLabels{2})
xlabel(sts.axesLabels{3})
%
%%% Plot colorband image:
figure()
image(permute(rgb,[1,3,2]))
title('Colormap in Colorbands')
ylabel('Colormap Index')
set(gca,'XTick',[], 'YTick',1:N, 'YDir','normal')
%%% Add colornames (if COLORNAMES is available):
try %#ok<TRYNC>
	text(ones(1,N), 1:N, colornames('CSS',rgb),...
		'HorizontalAlignment','center', 'BackgroundColor','white')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%maxdistcolor_demo