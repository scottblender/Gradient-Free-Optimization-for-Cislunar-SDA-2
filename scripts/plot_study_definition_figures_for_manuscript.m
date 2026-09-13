function outputs = plot_study_definition_figures_for_manuscript( ...
    inspectFigures,sections,outputDirectory)
%PLOT_STUDY_DEFINITION_FIGURES_FOR_MANUSCRIPT Use the proven definition renderer.
% The established renderer creates the study-definition figures. For the slot
% definition, the manuscript wrapper replaces the two legacy exports with
% versions containing inset zooms around the two marked adjacent slots.

if nargin<1 || isempty(inspectFigures), inspectFigures = false; end
if nargin<2 || isempty(sections), sections = "all"; end

paths = setup_project();
if nargin<3 || strlength(string(outputDirectory))==0
    outputDirectory = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDirectory = char(string(outputDirectory));
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

legacyDirectory = fullfile(paths.results,'study_definition_figures');
if isfolder(legacyDirectory), rmdir(legacyDirectory,'s'); end

requested = lower(string(sections(:)'));
available = ["catalog","slots","visibility","measurement","cases"];
if isequal(requested,"all"), requested = available; end
assert(all(ismember(requested,available)),'Unknown definition figure section.');
requested = unique(requested,'stable');

outputs = struct();
legacySections = requested(~ismember(requested,["visibility","slots"]));
if ~isempty(legacySections)
    transcript = evalc( ...
        'legacyOutputs = plot_study_definition_figures(inspectFigures,legacySections);');
    outputs = legacyOutputs;
end
if ismember("slots",requested)
    outputs.slots = create_zoomed_slot_definition( ...
        inspectFigures,legacyDirectory,paths);
end
if ismember("visibility",requested)
    outputs.visibilityGeometry = plot_visibility_keepout_geometry( ...
        inspectFigures,legacyDirectory);
end

outputs = relocate_output_paths(outputs,legacyDirectory,outputDirectory);

if isfolder(legacyDirectory)
    listing = dir(fullfile(legacyDirectory,'**','*'));
    listing = listing(~[listing.isdir]);
    for k = 1:numel(listing)
        move_one_file(fullfile(listing(k).folder,listing(k).name),outputDirectory);
    end
    if isfolder(legacyDirectory), rmdir(legacyDirectory,'s'); end
end

assert(~isfolder(legacyDirectory), ...
    'Legacy study-definition output directory should not remain after manuscript generation.');
fprintf('Study-definition manuscript files: %s\n',outputDirectory);
end


function outputs = create_zoomed_slot_definition(inspectFigure,outputDir,paths)
%CREATE_ZOOMED_SLOT_DEFINITION Export Figure 3 panels with readable insets.

if ~isfolder(outputDir), mkdir(outputDir); end
style = reviewer2_paper_style();
catalog = load(paths.catalog,'T');
T = catalog.T;
family = string(T.orbitFamily);
orbitIndex = find(family=="NNRHL1",1,'first');
assert(~isempty(orbitIndex),'No representative northern NRHO L1 orbit was found.');

periodAll = T.('Period (TU) ');
period = periodAll(orbitIndex);
rawTime = T.time{orbitIndex};
rawState = T.state{orbitIndex};
[uniqueTime,uniqueIndex] = unique(rawTime);
uniqueState = rawState(uniqueIndex,:);
interpolant = griddedInterpolant(uniqueTime,uniqueState,'spline');

numSlots = 50;
deltaTime = period/numSlots;
slotNumber = (1:numSlots).';
slotTime = (slotNumber-1)*deltaTime;
slotState = interpolant(slotTime);
selectedSlot = 17;
nextSlot = selectedSlot+1;
selectedColor = [0.85,0.25,0.20];
nextColor = [0.20,0.50,0.80];
orbitColor = [0.27,0.31,0.86];
neutralColor = [0.25,0.25,0.25];
mu = 1.215058560962404E-2;
LU = 384400;

% Orbit/slot geometry panel.
fig = manuscript_figure(style.geometryFigureWidth,style.geometryFigureHeight,style);
ax = axes(fig,'Units','normalized','Position',[0.10 0.12 0.78 0.67]);
prepare_3d_axes(ax,style);
plotStep = max(1,round(size(rawState,1)/500));
hOrbit = plot3(ax,rawState(1:plotStep:end,1),rawState(1:plotStep:end,2), ...
    rawState(1:plotStep:end,3),'-','Color',orbitColor,'LineWidth',2.5);
hSlots = plot3(ax,slotState(:,1),slotState(:,2),slotState(:,3),'o', ...
    'MarkerSize',5,'MarkerFaceColor','w','MarkerEdgeColor',neutralColor,'LineWidth',1.0);
hSelected = slot_marker(ax,slotState(selectedSlot,:),selectedColor,'o');
hNext = slot_marker(ax,slotState(nextSlot,:),nextColor,'s');
hMoon = draw_moon_local(ax,mu,LU);
axis(ax,'tight'); axis(ax,'vis3d');
format_axes(ax,style);
lgd = legend(ax,[hOrbit,hSlots,hSelected,hNext,hMoon], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1','Moon'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3,'Box','off');
format_legend(lgd,style);

localStart = max(1,selectedSlot-1);
localEnd = min(numSlots,nextSlot+1);
localTime = linspace(slotTime(localStart),slotTime(localEnd),160).';
localState = interpolant(localTime);
inset = axes(fig,'Units','normalized','Position',[0.28 0.23 0.38 0.36]);
inset.PositionConstraint = 'innerposition';
prepare_3d_axes(inset,style);
plot3(inset,localState(:,1),localState(:,2),localState(:,3),'-', ...
    'Color',orbitColor,'LineWidth',2.1,'HandleVisibility','off');
plot3(inset,slotState(localStart:localEnd,1),slotState(localStart:localEnd,2), ...
    slotState(localStart:localEnd,3),'o','MarkerSize',4,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9,'HandleVisibility','off');
slot_marker(inset,slotState(selectedSlot,:),selectedColor,'o');
slot_marker(inset,slotState(nextSlot,:),nextColor,'s');
set_local_limits(inset,[localState(:,1:3);slotState(localStart:localEnd,1:3)],0.18);
set(inset,'Box','on','FontSize',max(10,style.fontSize-2));
xlabel(inset,''); ylabel(inset,''); zlabel(inset,'');
inset.Position = [0.28 0.23 0.38 0.36];
text(inset,0.04,1.08,'Zoom','Units','normalized','FontName',style.fontName, ...
    'FontSize',style.fontSize-1,'FontWeight','bold','VerticalAlignment','bottom', ...
    'Clipping','off');

geometryFile = fullfile(outputDir,'slot_geometry_equal_time.eps');
export_figure(fig,geometryFile,inspectFigure,style);

% Normalized phase panel.
fig = manuscript_figure(style.slotPhaseFigureWidth,style.slotPhaseFigureHeight,style);
ax = axes(fig,'Units','normalized','Position',[0.11 0.18 0.82 0.60]);
hold(ax,'on'); box(ax,'off'); grid(ax,'off');
phase = slotTime/period;
plot(ax,[0,1],[0,0],'-','Color',0.65*[1,1,1],'LineWidth',1.5);
hCandidate = scatter(ax,phase,zeros(size(phase)),32,'w','filled', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9);
hSelectedPhase = scatter(ax,phase(selectedSlot),0,90,selectedColor,'filled', ...
    'MarkerEdgeColor','k','LineWidth',1.1);
hNextPhase = scatter(ax,phase(nextSlot),0,90,nextColor,'s','filled', ...
    'MarkerEdgeColor','k','LineWidth',1.1);
hEndpoint = plot(ax,1,0,'o','MarkerSize',9,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75,0.20,0.20],'LineWidth',1.8);
plot(ax,phase([selectedSlot,nextSlot]),[0.16,0.16],'-k','LineWidth',1.5);
plot(ax,phase([selectedSlot,selectedSlot]),[0,0.16],':k');
plot(ax,phase([nextSlot,nextSlot]),[0,0.16],':k');
text(ax,mean(phase([selectedSlot,nextSlot])),0.20,'\Delta t/T=1/50', ...
    'HorizontalAlignment','center','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');
text(ax,0.99,-0.025,{'t=T','not stored'},'HorizontalAlignment','right', ...
    'VerticalAlignment','top','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');
xlabel(ax,'Normalized epoch, t/T','FontWeight','bold');
yticks(ax,[]); ylim(ax,[-0.18,0.30]); xlim(ax,[-0.02,1.02]);
format_axes(ax,style);
lgd = legend(ax,[hCandidate,hSelectedPhase,hNextPhase,hEndpoint], ...
    {'Candidate slots','Slot j','Slot j+1','Excluded endpoint'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',2,'Box','off');
format_legend(lgd,style);

phaseInset = axes(fig,'Units','normalized','Position',[0.57 0.43 0.35 0.27]);
hold(phaseInset,'on'); box(phaseInset,'on'); grid(phaseInset,'off');
localIndex = max(1,selectedSlot-2):min(numSlots,nextSlot+2);
plot(phaseInset,phase(localIndex),zeros(size(localIndex)),'o-', ...
    'Color',neutralColor,'MarkerFaceColor','w','MarkerSize',5,'LineWidth',1.0);
scatter(phaseInset,phase(selectedSlot),0,82,selectedColor,'filled', ...
    'MarkerEdgeColor','k','LineWidth',1.0);
scatter(phaseInset,phase(nextSlot),0,82,nextColor,'s','filled', ...
    'MarkerEdgeColor','k','LineWidth',1.0);
plot(phaseInset,phase([selectedSlot,nextSlot]),[0.105,0.105],'-k','LineWidth',1.3);
plot(phaseInset,phase([selectedSlot,selectedSlot]),[0,0.105],':k');
plot(phaseInset,phase([nextSlot,nextSlot]),[0,0.105],':k');
text(phaseInset,mean(phase([selectedSlot,nextSlot])),0.125,'1/50', ...
    'HorizontalAlignment','center','FontName',style.fontName, ...
    'FontSize',style.fontSize-2,'FontWeight','bold');
pad = 0.35/numSlots;
xlim(phaseInset,[phase(localIndex(1))-pad,phase(localIndex(end))+pad]);
ylim(phaseInset,[-0.06,0.16]); yticks(phaseInset,[]);
set(phaseInset,'FontName',style.fontName,'FontSize',style.fontSize-2, ...
    'FontWeight','bold','LineWidth',style.axisLineWidth,'TickDir','out');
text(phaseInset,0.04,0.92,'Zoom','Units','normalized','FontName',style.fontName, ...
    'FontSize',style.fontSize-1,'FontWeight','bold','VerticalAlignment','top');

phaseFile = fullfile(outputDir,'slot_phase_grid.eps');
export_figure(fig,phaseFile,inspectFigure,style);

nextPosition = [slotState(2:end,1:3);slotState(1,1:3)];
adjacentChord_km = vecnorm(nextPosition-slotState(:,1:3),2,2)*LU;
orbitID = "";
if ismember('orbitID',T.Properties.VariableNames), orbitID = string(T.orbitID(orbitIndex)); end
slotSummary = table(orbitIndex,orbitID,family(orbitIndex),numSlots,period,deltaTime, ...
    min(adjacentChord_km),median(adjacentChord_km),max(adjacentChord_km), ...
    'VariableNames',{'catalogRow','orbitID','family','numSlots','period_TU', ...
    'deltaTime_TU','minimumChord_km','medianChord_km','maximumChord_km'});
summaryFile = fullfile(outputDir,'slot_definition_summary.csv');
writetable(slotSummary,summaryFile);

outputs = struct();
outputs.figures = [string(geometryFile);string(phaseFile)];
outputs.geometryFigure = string(geometryFile);
outputs.phaseFigure = string(phaseFile);
outputs.summary = string(summaryFile);
outputs.slotSummary = slotSummary;
end


function fig = manuscript_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize);
end


function prepare_3d_axes(ax,style)
hold(ax,'on'); grid(ax,'off'); box(ax,'off'); axis(ax,'equal');
view(ax,style.geometryAzimuth,style.geometryElevation);
ax.Projection = style.geometryProjection;
xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
end


function format_axes(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top', ...
    'XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
ax.ZLabel.FontSize = style.labelFontSize;
end


function format_legend(lgd,style)
lgd.FontName = style.fontName;
lgd.FontSize = style.legendFontSize;
lgd.FontWeight = 'bold';
lgd.ItemTokenSize = style.legendItemTokenSize;
end


function h = slot_marker(ax,state,color,marker)
h = plot3(ax,state(1),state(2),state(3),marker,'MarkerSize',13, ...
    'MarkerFaceColor','w','MarkerEdgeColor',color,'LineWidth',2.8);
if marker=='o', inner='+'; else, inner='x'; end
plot3(ax,state(1),state(2),state(3),inner,'MarkerSize',9,'Color','k', ...
    'LineWidth',1.8,'HandleVisibility','off');
end


function h = draw_moon_local(ax,mu,LU)
radius = 1737.1/LU;
[x,y,z] = sphere(24);
h = surf(ax,radius*x+1-mu,radius*y,radius*z,'FaceColor',[0.72,0.72,0.72], ...
    'EdgeColor','none','FaceLighting','gouraud');
camlight(ax,'headlight'); material(ax,'dull');
end


function set_local_limits(ax,data,padFraction)
for dimension=1:3
    values=data(:,dimension); lo=min(values); hi=max(values); span=max(hi-lo,1e-5);
    limits=[lo hi]+[-1 1]*padFraction*span;
    if dimension==1, xlim(ax,limits); elseif dimension==2, ylim(ax,limits); else, zlim(ax,limits); end
end
axis(ax,'vis3d');
end


function export_figure(fig,fileName,inspectFigure,style)
if inspectFigure, figure(fig); drawnow; pause(5); end
drawnow; finalize_manuscript_figure(fig);
print(fig,fileName,'-depsc2','-painters','-r600','-loose');
[folder,stem] = fileparts(fileName);
exportgraphics(fig,fullfile(folder,[stem '.png']),'Resolution',style.exportDpi);
close(fig);
end


function value = relocate_output_paths(value,legacyDirectory,outputDirectory)
if isstruct(value)
    fields = fieldnames(value);
    for k = 1:numel(value)
        for f = 1:numel(fields)
            value(k).(fields{f}) = relocate_output_paths( ...
                value(k).(fields{f}),legacyDirectory,outputDirectory);
        end
    end
    return;
end
if iscell(value)
    for k = 1:numel(value)
        value{k} = relocate_output_paths(value{k},legacyDirectory,outputDirectory);
    end
    return;
end
if isstring(value)
    for k = 1:numel(value)
        candidate = value(k);
        if strlength(candidate)>0 && isfile(candidate) && ...
                startsWith(candidate,string(legacyDirectory),'IgnoreCase',true)
            value(k) = string(move_one_file(char(candidate),outputDirectory));
        end
    end
    return;
end
if ischar(value) && isrow(value) && isfile(value) && ...
        startsWith(string(value),string(legacyDirectory),'IgnoreCase',true)
    value = move_one_file(value,outputDirectory);
end
end


function destination = move_one_file(source,outputDirectory)
[~,name,extension] = fileparts(source);
destination = fullfile(outputDirectory,[name extension]);
if ~strcmpi(source,destination)
    if isfile(destination), delete(destination); end
    [ok,message] = movefile(source,destination);
    assert(ok,'Could not move manuscript definition file %s: %s',source,message);
end
if strcmpi(extension,'.eps')
    sourcePng = fullfile(fileparts(source),[name '.png']);
    destinationPng = fullfile(outputDirectory,[name '.png']);
    if isfile(sourcePng) && ~strcmpi(sourcePng,destinationPng)
        if isfile(destinationPng), delete(destinationPng); end
        [ok,message] = movefile(sourcePng,destinationPng);
        assert(ok,'Could not move manuscript PNG preview %s: %s',sourcePng,message);
    end
end
end