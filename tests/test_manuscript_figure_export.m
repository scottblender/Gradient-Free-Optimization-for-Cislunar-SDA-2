function test_manuscript_figure_export()
% Verify full content centering and a non-mutating writer; no catalog required.
setup_project(); style=reviewer2_paper_style();
folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s'));
fig=figure('Visible','off','Color','w','Units','inches', ...
    'Position',[1 1 style.figureWidth style.figureHeight], ...
    'PaperUnits','inches','PaperSize',[style.figureWidth style.figureHeight], ...
    'PaperPosition',[0 0 style.figureWidth style.figureHeight], ...
    'PaperPositionMode','manual','Renderer','painters');
closeFigure=onCleanup(@() close(fig));
ax=axes(fig,'Position',style.geometryPlotPosition,'FontSize',style.fontSize,'FontWeight','bold');
theta=linspace(0,2*pi,401);
plot3(ax,1+0.1*cos(theta),0.1*sin(theta),0.2*cos(theta));
axis(ax,'equal'); view(ax,-37.5,30); ax.Projection='perspective';
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
lgd=legend(ax,'Orbit','Location','northoutside','FontWeight','bold'); drawnow;
format_manuscript_legend(ax,lgd,style,style.geometryPlotPosition);
assert(strcmp(lgd.FontWeight,'bold') && lgd.FontSize==style.geometryLegendFontSize);

% Measure the final rendered content in figure pixels. The legend itself must
% be figure-centered horizontally, and the union of legend + axes/ticks/labels
% must have equal left/right AND top/bottom margins on the export canvas.
figUnits=fig.Units; axUnits=ax.Units; legendUnits=lgd.Units;
fig.Units='pixels'; ax.Units='pixels'; lgd.Units='pixels'; drawnow;
figPosition=fig.Position; axesPosition=ax.Position; inset=ax.TightInset; legendPosition=lgd.Position;
legendCenter=legendPosition(1)+0.5*legendPosition(3);
assert(abs(legendCenter-0.5*figPosition(3)) <= 1, ...
    'Legend is not centered on the figure bounding box.');
contentLeft=min(axesPosition(1)-inset(1),legendPosition(1));
contentRight=max(axesPosition(1)+axesPosition(3)+inset(3), ...
    legendPosition(1)+legendPosition(3));
contentBottom=min(axesPosition(2)-inset(2),legendPosition(2));
contentTop=max(axesPosition(2)+axesPosition(4)+inset(4), ...
    legendPosition(2)+legendPosition(4));
leftMargin=contentLeft;
rightMargin=figPosition(3)-contentRight;
bottomMargin=contentBottom;
topMargin=figPosition(4)-contentTop;
assert(abs(leftMargin-rightMargin) <= 1, ...
    'Final visible content does not have equal left/right margins.');
assert(abs(bottomMargin-topMargin) <= 1, ...
    'Final visible content does not have equal top/bottom margins.');
fig.Units=figUnits; ax.Units=axUnits; lgd.Units=legendUnits; drawnow;

properties={'Position','XLim','YLim','ZLim','XTick','YTick','ZTick', ...
    'FontSize','FontWeight','View','Projection','DataAspectRatio','Clipping'};
before=cellfun(@(p) get(ax,p),properties,'UniformOutput',false);
legendBefore=lgd.Position; paperBefore=fig.PaperPosition;
file=fullfile(folder,'scene.eps'); export_manuscript_figure(fig,file);
after=cellfun(@(p) get(ax,p),properties,'UniformOutput',false);
assert(isequaln(before,after),'Export reformatted the scene.');
assert(isequal(legendBefore,lgd.Position) && isequal(paperBefore,fig.PaperPosition));
assert(isfile(file) && isfile(fullfile(folder,'scene.png')));
assert(startsWith(fileread(file),'%!PS-Adobe'));
fprintf('Full content-centering and non-mutating export checks passed.\n');
end
