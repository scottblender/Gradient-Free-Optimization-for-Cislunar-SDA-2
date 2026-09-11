function test_manuscript_figure_export()
% Data-free 3-D regression for the clipped Halo export reported in review.
setup_project(); style=reviewer2_paper_style();
folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes=strings(2,1); imageSizes=zeros(2,2);
for k=1:2
    fig=figure('Visible','off','Color','w','Units','inches', ...
        'Position',[1 1 style.geometryFigureWidth style.geometryFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.geometryFigureWidth style.geometryFigureHeight]);
    closeFigure=onCleanup(@() close(fig));
    ax=axes(fig); hold(ax,'on'); theta=linspace(0,2*pi,401);
    handles=gobjects(5,1);
    for j=1:2
        handles(j)=plot3(ax,0.8+0.3*j+0.08*cos(theta), ...
            0.1*sin(theta),0.15*cos(theta),'LineWidth',1);
    end
    for j=3:5, handles(j)=plot3(ax,0.8+0.1*j,0,0,'o'); end
    axis(ax,'equal'); axis(ax,'tight'); axis(ax,'vis3d');
    view(ax,-37.5,30); ax.Projection='perspective';
    xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
    legend(ax,handles,{'L1','L2','Moon','L1 point','L2 point'},'Location','northoutside');
    coordinates=[handles(1).XData;handles(1).YData;handles(1).ZData];
    limits=[ax.XLim ax.YLim ax.ZLim]; direction=ax.View; aspect=ax.DataAspectRatio;
    prepare_manuscript_figure(fig);
    assert(strcmp(ax.Clipping,'off') && strcmp(handles(1).Clipping,'off'));
    assert(isequal(coordinates,[handles(1).XData;handles(1).YData;handles(1).ZData]));
    assert(isequal(limits,[ax.XLim ax.YLim ax.ZLim]) && isequal(direction,ax.View));
    assert(isequal(aspect,ax.DataAspectRatio) && strcmp(ax.Projection,'perspective'));
    ax.Legend.Units='normalized'; lp=ax.Legend.Position; ap=ax.Position;
    assert(lp(1)>=0 && lp(1)+lp(3)<=1 && lp(2)>ap(2)+ap(4));
    fonts=findall(fig,'-property','FontSize');
    for j=1:numel(fonts)
        if isprop(fonts(j),'FontWeight'), assert(strcmp(fonts(j).FontWeight,'bold')); end
    end
    beforePosition=ax.Position; beforeTicks={ax.XTick,ax.YTick,ax.ZTick};
    file=fullfile(folder,sprintf('halo_%d.eps',k));
    export_manuscript_figure(fig,file);
    assert(isequal(beforePosition,ax.Position) && isequal(beforeTicks,{ax.XTick,ax.YTick,ax.ZTick}), ...
        'Export changed the prepared scene.');
    epsText=fileread(file);
    boxes(k)=string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
    assert(strlength(boxes(k))>0);
    png=imfinfo(strrep(file,'.eps','.png')); imageSizes(k,:)=[png.Width png.Height];
    clear closeFigure;
end
assert(boxes(1)==boxes(2),'Equal geometry panels have different EPS canvases.');
assert(isequal(imageSizes(1,:),imageSizes(2,:)),'Equal geometry panels have different PNG canvases.');
fprintf('Manuscript 3-D export checks passed.\n');
end
