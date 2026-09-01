function outputs = plot_slot_definition(inspectFigure)
% Illustrate the exact equal-time, endpoint-excluded 50-slot definition.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

catalog = load(projectPaths.catalog,'T');
T = catalog.T;

family = string(T.orbitFamily);
orbitIndex = find(family=="NNRHL1",1,'first');
if isempty(orbitIndex)
    orbitIndex = find(family=="NHL1",1,'first');
end
assert(~isempty(orbitIndex), ...
    'No representative northern L1 orbit was found.');

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

assert(slotTime(1)==0,'The first slot must occur at t=0.');
assert(abs(slotTime(end)-49*period/50)<=10*eps(period), ...
    'The last slot must occur at 49T/50.');
assert(all(diff(slotTime)>0),'Slot epochs are not strictly increasing.');
assert(all(slotTime<period),'The periodic endpoint must not be stored.');

nextPosition = [slotState(2:end,1:3);slotState(1,1:3)];
adjacentChord_km = vecnorm( ...
    nextPosition-slotState(:,1:3),2,2)*384400;

selectedSlot = 17;
nextSlot = selectedSlot+1;
selectedColor = [0.85,0.25,0.20];
nextColor = [0.20,0.50,0.80];
orbitColor = [0.27,0.31,0.86];
neutralColor = [0.25,0.25,0.25];

fig = figure('Color','w','Units','inches', ...
    'Position',[1,1,12,4.2], ...
    'PaperUnits','inches','PaperPosition',[0,0,12,4.2]);
layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

% (a) Geometric placement of all 50 equal-time samples.
ax1 = nexttile(layout);
hold(ax1,'on'); box(ax1,'on'); axis(ax1,'equal');
plotStep = max(1,round(size(rawState,1)/500));
plot3(ax1,rawState(1:plotStep:end,1), ...
    rawState(1:plotStep:end,2),rawState(1:plotStep:end,3), ...
    '-','Color',orbitColor,'LineWidth',1.8);
plot3(ax1,slotState(:,1),slotState(:,2),slotState(:,3), ...
    'o','MarkerSize',4.5,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9);
plot3(ax1,slotState(selectedSlot,1),slotState(selectedSlot,2), ...
    slotState(selectedSlot,3),'o','MarkerSize',8, ...
    'MarkerFaceColor',selectedColor,'MarkerEdgeColor','k');
plot3(ax1,slotState(nextSlot,1),slotState(nextSlot,2), ...
    slotState(nextSlot,3),'s','MarkerSize',8, ...
    'MarkerFaceColor',nextColor,'MarkerEdgeColor','k');
xlabel(ax1,'x (LU)'); ylabel(ax1,'y (LU)'); zlabel(ax1,'z (LU)');
title(ax1,'(a) Equal-time states');
view(ax1,32,24);

% (b) Exact normalized phase convention, including endpoint exclusion.
ax2 = nexttile(layout);
hold(ax2,'on'); box(ax2,'on');
phase = slotTime/period;
plot(ax2,[0,1],[0,0],'-','Color',0.65*[1,1,1],'LineWidth',1.2);
scatter(ax2,phase,zeros(size(phase)),24,'w','filled', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.8);
scatter(ax2,phase(selectedSlot),0,70,selectedColor,'filled', ...
    'MarkerEdgeColor','k');
scatter(ax2,phase(nextSlot),0,70,nextColor,'s','filled', ...
    'MarkerEdgeColor','k');
plot(ax2,1,0,'o','MarkerSize',7,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75,0.20,0.20],'LineWidth',1.5);
plot(ax2,phase([selectedSlot,nextSlot]),[0.13,0.13],'-k','LineWidth',1.2);
plot(ax2,phase([selectedSlot,selectedSlot]),[0,0.13],':k');
plot(ax2,phase([nextSlot,nextSlot]),[0,0.13],':k');
text(ax2,mean(phase([selectedSlot,nextSlot])),0.16, ...
    '\Delta t/T=1/50','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontWeight','bold');
text(ax2,0.99,-0.15,{'t=T','not stored'}, ...
    'HorizontalAlignment','right','VerticalAlignment','top', ...
    'FontName','Times New Roman','FontWeight','bold');
xlabel(ax2,'Normalized epoch, t/T');
yticks(ax2,[]); ylim(ax2,[-0.28,0.28]); xlim(ax2,[-0.02,1.02]);
title(ax2,'(b) Endpoint-excluded phase grid');

% (c) Equal time does not imply equal arc length.
ax3 = nexttile(layout);
hold(ax3,'on'); box(ax3,'on'); grid(ax3,'on');
plot(ax3,slotNumber,adjacentChord_km,'-o', ...
    'Color',orbitColor,'MarkerFaceColor',orbitColor, ...
    'MarkerSize',3.5,'LineWidth',1.5);
plot(ax3,selectedSlot,adjacentChord_km(selectedSlot),'o', ...
    'MarkerSize',8,'MarkerFaceColor',selectedColor,'MarkerEdgeColor','k');
yline(ax3,median(adjacentChord_km),'--','Median', ...
    'Color',0.35*[1,1,1],'LabelHorizontalAlignment','left');
xlabel(ax3,'Slot j');
ylabel(ax3,'Chord distance j to j+1 (km)');
xlim(ax3,[1,numSlots]);
title(ax3,'(c) Unequal spatial separation');

allAxes = [ax1,ax2,ax3];
set(allAxes,'FontName','Times New Roman','FontSize',11, ...
    'FontWeight','bold','LineWidth',1.2,'TickLabelInterpreter','tex');

figureFile = fullfile(outputDir,'equal_time_slot_definition.eps');
inspect_before_export(fig,inspectFigure,'equal-time slot-definition');
exportgraphics(fig,figureFile,'ContentType','image','Resolution',600);
close(fig);

orbitID = "";
if ismember('orbitID',T.Properties.VariableNames)
    orbitID = string(T.orbitID(orbitIndex));
end

slotSummary = table( ...
    orbitIndex,orbitID,family(orbitIndex),numSlots,period,deltaTime, ...
    min(adjacentChord_km),median(adjacentChord_km),max(adjacentChord_km), ...
    'VariableNames',{'catalogRow','orbitID','family','numSlots', ...
    'period_TU','deltaTime_TU','minimumChord_km', ...
    'medianChord_km','maximumChord_km'});

summaryFile = fullfile(outputDir,'slot_definition_summary.csv');
writetable(slotSummary,summaryFile);

outputs = struct();
outputs.figure = string(figureFile);
outputs.summary = string(summaryFile);
outputs.slotSummary = slotSummary;

fprintf('Saved equal-time slot-definition figure to:\n  %s\n',figureFile);
end


function inspect_before_export(fig,inspectFigure,description)

if inspectFigure
    figure(fig);
    drawnow;
    input(sprintf( ...
        'Inspect the %s figure, then press Enter to export: ', ...
        description),'s');
end
end
