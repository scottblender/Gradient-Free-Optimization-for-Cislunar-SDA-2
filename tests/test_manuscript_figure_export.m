function test_manuscript_figure_export()
% Data-free regression for paired export canvases and manuscript typography.
paths = setup_project(); %#ok<NASGU>
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes = strings(2,1); imageSizes = zeros(2,2);
for k = 1:2
    fig = figure('Visible','off','Units','inches', ...
        'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
    closeFigure = onCleanup(@() close(fig));
    ax = axes(fig); plot(ax,1:10,k*(1:10)); xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)');
    text(ax,5,5*k,repmat('label ',1,k),'FontSize',8);
    file = fullfile(folder,sprintf('panel%d.eps',k));
    meta = export_manuscript_figure(fig,file);
    epsText = fileread(file);
    boxes(k) = string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
    assert(contains(epsText,'%%HiResBoundingBox: 0 0 468.000000 374.400000'));
    assert(meta.minimumPrintedFontPoints>=style.minimumPrintedFontSize);
    png = imfinfo(strrep(file,'.eps','.png')); imageSizes(k,:)=[png.Width png.Height];
    objects=findall(fig,'-property','FontSize');
    assert(all(arrayfun(@(obj) obj.FontSize>=style.fontSize,objects)));
    clear closeFigure;
end
assert(boxes(1)==boxes(2),'Paired EPS canvases differ.');
assert(isequal(imageSizes(1,:),imageSizes(2,:)),'Paired PNG canvases differ.');
assert(style.geometryFigureWidth==style.measurementFigureWidth && ...
    style.geometryFigureHeight==style.measurementFigureHeight);
fprintf('Manuscript EPS canvas and font checks passed.\n');
end
