function outputs = plot_orbit_catalog_manuscript(inspectFigure,outputDir)
%PLOT_ORBIT_CATALOG_MANUSCRIPT Final representative observer-orbit figures.
% Explicit limits are computed from the plotted data so no 3-D layout or
% clipping state is changed by the exporter.

if nargin<1 || isempty(inspectFigure), inspectFigure=false; end
paths=setup_project();
if nargin<2 || strlength(string(outputDir))==0
    outputDir=fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDir=char(string(outputDir)); if ~isfolder(outputDir), mkdir(outputDir); end
style=reviewer2_paper_style();

S=load(paths.catalog,'T'); T=S.T;
required={'orbitFamily','state','Period (TU) ','Stability index  '};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'The orbit catalog is missing required plotting variables.');
LU=384400; TU=375695; Rmoon=1737.1; mu=1.215058560962404E-2; xMoon=1-mu;
familyOrder=["NHL1","NHL2","SHL1","SHL2","NNRHL1","NNRHL2","SNRHL1","SNRHL2","DRO"];
family=string(T.orbitFamily);
assert(all(ismember(family,familyOrder)),'The catalog contains an unexpected orbit family.');

orbitID=strings(height(T),1);
if ismember('Id',T.Properties.VariableNames)
    orbitID=string(T.Id);
elseif ismember('orbitID',T.Properties.VariableNames)
    orbitID=string(T.orbitID);
end

% Quantitative catalog products used by the manuscript table.
nOrbit=height(T);
periluneAltitude_km=zeros(nOrbit,1); apoluneAltitude_km=zeros(nOrbit,1);
inPlaneAmplitude_km=zeros(nOrbit,1); outOfPlaneAmplitude_km=zeros(nOrbit,1);
for k=1:nOrbit
    state=double(T.state{k}); pos=state(:,1:3); rel=pos-[xMoon 0 0];
    d=vecnorm(rel,2,2)*LU;
    periluneAltitude_km(k)=min(d)-Rmoon;
    apoluneAltitude_km(k)=max(d)-Rmoon;
    dx=max(pos(:,1))-min(pos(:,1)); dy=max(pos(:,2))-min(pos(:,2));
    dz=max(pos(:,3))-min(pos(:,3));
    inPlaneAmplitude_km(k)=0.5*hypot(dx,dy)*LU;
    outOfPlaneAmplitude_km(k)=0.5*dz*LU;
end
period_TU=double(T.('Period (TU) ')); period_days=period_TU*TU/86400;
stabilityIndex=double(T.('Stability index  '));
orbitMetrics=table(orbitID,family,periluneAltitude_km,apoluneAltitude_km,period_TU, ...
    period_days,stabilityIndex,inPlaneAmplitude_km,outOfPlaneAmplitude_km);
metricFile=fullfile(outputDir,'orbit_catalog_metrics.csv'); writetable(orbitMetrics,metricFile);

nFamily=numel(familyOrder); Family=familyOrder(:);
lagrangePoint=["L1";"L2";"L1";"L2";"L1";"L2";"L1";"L2";"Moon-centered"];
count=zeros(nFamily,1); periluneMin_km=zeros(nFamily,1); periluneMax_km=zeros(nFamily,1);
apoluneMin_km=zeros(nFamily,1); apoluneMax_km=zeros(nFamily,1);
periodMin_TU=zeros(nFamily,1); periodMax_TU=zeros(nFamily,1);
periodMin_days=zeros(nFamily,1); periodMax_days=zeros(nFamily,1);
stabilityMin=zeros(nFamily,1); stabilityMax=zeros(nFamily,1);
inPlaneMin_km=zeros(nFamily,1); inPlaneMax_km=zeros(nFamily,1);
outOfPlaneMin_km=zeros(nFamily,1); outOfPlaneMax_km=zeros(nFamily,1);
for k=1:nFamily
    use=family==familyOrder(k); count(k)=nnz(use); assert(count(k)>0);
    periluneMin_km(k)=min(periluneAltitude_km(use)); periluneMax_km(k)=max(periluneAltitude_km(use));
    apoluneMin_km(k)=min(apoluneAltitude_km(use)); apoluneMax_km(k)=max(apoluneAltitude_km(use));
    periodMin_TU(k)=min(period_TU(use)); periodMax_TU(k)=max(period_TU(use));
    periodMin_days(k)=min(period_days(use)); periodMax_days(k)=max(period_days(use));
    stabilityMin(k)=min(stabilityIndex(use)); stabilityMax(k)=max(stabilityIndex(use));
    inPlaneMin_km(k)=min(inPlaneAmplitude_km(use)); inPlaneMax_km(k)=max(inPlaneAmplitude_km(use));
    outOfPlaneMin_km(k)=min(outOfPlaneAmplitude_km(use)); outOfPlaneMax_km(k)=max(outOfPlaneAmplitude_km(use));
end
familySummary=table(Family,lagrangePoint,count,periluneMin_km,periluneMax_km, ...
    apoluneMin_km,apoluneMax_km,periodMin_TU,periodMax_TU,periodMin_days, ...
    periodMax_days,stabilityMin,stabilityMax,inPlaneMin_km,inPlaneMax_km, ...
    outOfPlaneMin_km,outOfPlaneMax_km);
summaryFile=fullfile(outputDir,'orbit_family_summary.csv'); writetable(familySummary,summaryFile);

familyGroups={ ["NHL1","NHL2"],["SHL1","SHL2"], ...
    ["NNRHL1","NNRHL2"],["SNRHL1","SNRHL2"],["DRO"] };
figureNames=["northern_halo","southern_halo","northern_rectilinear", ...
    "southern_rectilinear","dro_family"];
[xL1,xL2]=cr3bp_L1L2(mu);
cL1=[0.05 0.32 0.82]; cL2=[0.90 0.16 0.12]; cMoon=[0.70 0.70 0.70]; cPoint=[0.85 0.85 0.85];
figureFiles=strings(numel(familyGroups),1);

for g=1:numel(familyGroups)
    group=familyGroups{g}; key=char(figureNames(g));
    fig=manuscript_figure(style.orbitFamilyFigureWidth,style.orbitFamilyFigureHeight,style);
    ax=axes(fig,'Units','normalized','Position',style.geometryPlotPosition);
    ax.PositionConstraint='innerposition'; hold(ax,'on'); box(ax,'off'); grid(ax,'off'); axis(ax,'equal');
    viewAngles=style.orbitFamilyViews.(key); view(ax,viewAngles(1),viewAngles(2));
    ax.Projection=style.orbitFamilyProjections.(key); style_axes(ax,style);
    allPoints=zeros(0,3);

    if numel(group)==2
        colors=[cL1;cL2]; familyHandles=gobjects(2,1); familyLabels=["L1";"L2"];
        for member=1:2
            rows=find(family==group(member)); assert(numel(rows)==50);
            selected=rows(unique(round(linspace(1,numel(rows),16))));
            for j=1:numel(selected)
                state=double(T.state{selected(j)}); step=max(1,round(size(state,1)/300));
                p=state(1:step:end,1:3); allPoints=[allPoints;p]; %#ok<AGROW>
                h=plot3(ax,p(:,1),p(:,2),p(:,3),'-','Color',colors(member,:), ...
                    'LineWidth',0.85,'Clipping','off');
                if j==1, familyHandles(member)=h; else, h.HandleVisibility='off'; end
            end
        end
        moonHandle=plot3(ax,xMoon,0,0,'o','MarkerSize',6,'MarkerFaceColor',cMoon, ...
            'MarkerEdgeColor',[0.45 0.45 0.45],'LineWidth',0.9,'Clipping','off');
        l1Handle=plot3(ax,xL1,0,0,'^','MarkerSize',7,'MarkerFaceColor',cPoint, ...
            'MarkerEdgeColor',[0.55 0.55 0.55],'LineWidth',0.9,'Clipping','off');
        l2Handle=plot3(ax,xL2,0,0,'v','MarkerSize',7,'MarkerFaceColor',cPoint, ...
            'MarkerEdgeColor',[0.55 0.55 0.55],'LineWidth',0.9,'Clipping','off');
        allPoints=[allPoints;xMoon 0 0;xL1 0 0;xL2 0 0];
        xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
        apply_explicit_limits(ax,allPoints,style);
        axis(ax,'vis3d'); simplify_geometry_ticks(ax);
        lgd=legend(ax,[familyHandles;moonHandle;l1Handle;l2Handle], ...
            cellstr([familyLabels;"Moon";"L1 point";"L2 point"]), ...
            'Location','northoutside','Orientation','horizontal','NumColumns',5,'Box','off');
        style_legend(lgd,style,[14 8]); place_legend(ax,lgd,style.geometryPlotPosition,style);
    else
        rows=find(family=="DRO"); assert(numel(rows)==50);
        selected=unique(round(linspace(1,numel(rows),16))); droHandle=gobjects(1); allXY=zeros(0,2);
        for j=1:numel(selected)
            state=double(T.state{rows(selected(j))}); step=max(1,round(size(state,1)/300));
            p=state(1:step:end,1:2); allXY=[allXY;p]; %#ok<AGROW>
            h=plot(ax,p(:,1),p(:,2),'-','Color',cL1,'LineWidth',1.45,'Clipping','off');
            if j==1, droHandle=h; else, h.HandleVisibility='off'; end
        end
        moonRadius=Rmoon/LU; a=linspace(0,2*pi,200);
        fill(ax,xMoon+moonRadius*cos(a),moonRadius*sin(a),cMoon,'EdgeColor','none','HandleVisibility','off');
        moonHandle=plot(ax,xMoon,0,'o','MarkerSize',3,'MarkerFaceColor',cMoon,'MarkerEdgeColor','k','LineWidth',0.8);
        xData=[allXY(:,1);xMoon+moonRadius*cos(a(:))]; yData=[allXY(:,2);moonRadius*sin(a(:))];
        xPad=0.05*max(max(xData)-min(xData),eps); yPad=0.05*max(max(yData)-min(yData),eps);
        xlim(ax,[min(xData)-xPad,max(xData)+xPad]); ylim(ax,[min(yData)-yPad,max(yData)+yPad]);
        xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)');
        lgd=legend(ax,[droHandle;moonHandle],{'DRO','Moon'},'Location','northoutside', ...
            'Orientation','horizontal','NumColumns',2,'Box','off');
        style_legend(lgd,style,[14 8]); place_legend(ax,lgd,style.geometryPlotPosition,style);
    end

    figureFiles(g)=fullfile(outputDir,figureNames(g)+".eps");
    if inspectFigure
        figure(fig); drawnow; fprintf('Previewing %s for 5 seconds before export.\n',figureNames(g)); pause(5);
    end
    drawnow; export_manuscript_figure(fig,figureFiles(g)); close(fig);
end

outputs=struct('figures',figureFiles,'familySummary',string(summaryFile), ...
    'orbitMetrics',string(metricFile),'numOrbits',nOrbit,'numFamilies',nFamily);
fprintf('Saved orbit-family manuscript figures and catalog tables to:\n  %s\n',outputDir);
end


function fig=manuscript_figure(w,h,style)
fig=figure('Color','w','Units','inches','Position',[1 1 w h],'PaperUnits','inches', ...
    'PaperPosition',[0 0 w h],'PaperSize',[w h],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize, ...
    'DefaultAxesFontWeight',style.fontWeight,'DefaultTextFontName',style.fontName, ...
    'DefaultTextFontSize',style.fontSize,'DefaultTextFontWeight',style.fontWeight);
end

function style_axes(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight',style.fontWeight, ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top','Box','off', ...
    'XGrid','off','YGrid','off','ZGrid','off','TickLabelInterpreter','tex');
end

function style_legend(lgd,style,itemToken)
lgd.FontName=style.fontName; lgd.FontSize=style.fontSize; lgd.FontWeight=style.fontWeight;
lgd.ItemTokenSize=itemToken; lgd.Box='off';
end

function place_legend(ax,lgd,plotPosition,style)
lgd.Location='northoutside'; lgd.Units='normalized'; drawnow; north=lgd.Position;
lgd.Location='none'; ax.PositionConstraint='innerposition'; ax.Position=plotPosition; drawnow;
pos=lgd.Position; pos(1)=max(0.002,min(0.5-pos(3)/2,0.998-pos(3)));
pos(2)=min(max(north(2)+style.legendNorthOutsideYOffset, ...
    plotPosition(2)+plotPosition(4)+style.legendMinimumGap),0.99-pos(4));
lgd.Position=pos; lgd.AutoUpdate='off'; ax.Position=plotPosition; drawnow;
end

function apply_explicit_limits(ax,points,style)
pad=[style.geometryXPadding style.geometryYPadding style.geometryZPadding];
for d=1:3
    v=points(:,d); lo=min(v); hi=max(v); span=hi-lo;
    if span<=100*eps(max(1,max(abs(v)))), span=max(0.02,0.05*max(1,abs(mean(v)))); end
    lim=[lo-pad(d)*span,hi+pad(d)*span];
    if d==1, xlim(ax,lim); elseif d==2, ylim(ax,lim); else, zlim(ax,lim); end
end
end

function simplify_geometry_ticks(ax)
for name=["X","Y","Z"]
    prop=name+"Tick"; ticks=double(ax.(prop));
    if numel(ticks)~=3 || any(~isfinite(ticks)), continue; end
    tol=100*eps(max(1,max(abs(ticks))));
    if abs(ticks(2))<=tol && abs(ticks(1)+ticks(3))<=tol, ax.(prop)=ticks([1 3]); end
end
end

function [xL1,xL2]=cr3bp_L1L2(mu)
f=@(x) x-(1-mu)*(x+mu)./abs(x+mu).^3-mu*(x-(1-mu))./abs(x-(1-mu)).^3;
d=(mu/3)^(1/3); xL1=fzero(f,[1-mu-d,1-mu-1e-6]); xL2=fzero(f,[1-mu+1e-6,1-mu+d+0.5]);
end
