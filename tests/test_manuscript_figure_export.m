function test_manuscript_figure_export()
% Verify the writer leaves a completed scene unchanged; no catalog required.
setup_project(); style=reviewer2_paper_style();
folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
fig=figure('Visible','off','Color','w','Units','inches', ...
    'Position',[1 1 style.figureWidth style.figureHeight], ...
    'PaperUnits','inches','PaperSize',[style.figureWidth style.figureHeight], ...
    'PaperPosition',[0 0 style.figureWidth style.figureHeight], ...
    'PaperPositionMode','manual','Renderer','painters');
closeFigure=onCleanup(@() close(fig)); %#ok<NASGU>
ax=axes(fig,'Position',style.geometryPlotPosition,'FontSize',style.fontSize,'FontWeight','bold');
theta=linspace(0,2*pi,401);
plot3(ax,1+0.1*cos(theta),0.1*sin(theta),0.2*cos(theta));
axis(ax,'equal'); view(ax,-37.5,30); ax.Projection='perspective';
xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
lgd=legend(ax,'Orbit','Location','northoutside','FontWeight','bold'); drawnow;
format_manuscript_legend(ax,lgd,style,style.geometryPlotPosition);
assert(strcmp(lgd.FontWeight,'bold') && lgd.FontSize==style.geometryLegendFontSize);
assert(abs((lgd.Position(1)+0.5*lgd.Position(3))-0.5) < 1e-10, ...
    '3-D legend is not centered on the common export canvas.');
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
fprintf('Non-mutating export checks passed.\n');
end
