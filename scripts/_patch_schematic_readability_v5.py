from pathlib import Path

path = Path('scripts/plot_study_definition_figures.m')
text = path.read_text()

vis_start = text.index('function outputs = create_visibility_keepout_figure')
meas_start = text.index('function outputs = create_measurement_model_figure')
helper_start = text.index('function [fig,axesHandles,textAx] = schematic_figure_layout')

prefix = text[:vis_start]
vis = text[vis_start:meas_start]
meas = text[meas_start:helper_start]
suffix = text[helper_start:]

vis_replacements = [
    ("draw_angle_arc_2d(ax,observer,0,thetaOcc,0.66,cOcc,1.5);\ndraw_angle_arc_2d(ax,observer,0,thetaKeepout,1.14,cSensor,1.7);\ndraw_angle_arc_2d(ax,observer,0,thetaTarget,1.72,cTarget,1.7);",
     "draw_angle_arc_2d(ax,observer,0,thetaOcc,0.78,cOcc,1.7);\ndraw_angle_arc_2d(ax,observer,0,thetaKeepout,1.34,cSensor,1.9);\ndraw_angle_arc_2d(ax,observer,0,thetaTarget,2.02,cTarget,1.9);",
     'angle arcs'),
    ("text(ax,observer(1),observer(2)-0.36,'Observer', ...\n    'Color',cObserver,'FontWeight','bold','FontSize',12, ...\n    'HorizontalAlignment','center');",
     "text(ax,observer(1)-0.06,observer(2)-0.44,'Observer', ...\n    'Color',cObserver,'FontWeight','bold','FontSize',14, ...\n    'HorizontalAlignment','center');",
     'observer label'),
    ("text(ax,target(1)+0.03,target(2)+0.20,'Target', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',12, ...\n    'HorizontalAlignment','center');",
     "text(ax,target(1)+0.16,target(2)-0.12,'Target', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',14, ...\n    'HorizontalAlignment','left');",
     'target label'),
    ("text(ax,body(1)+0.02,body(2)-0.82,'Body b', ...\n    'FontWeight','bold','FontSize',11,'HorizontalAlignment','center');",
     "text(ax,body(1)+0.02,body(2)-0.90,'Body b', ...\n    'FontWeight','bold','FontSize',13,'HorizontalAlignment','center');",
     'body label'),
    ("text(ax,observer(1)+0.77*cos(thetaOcc/2), ...\n    observer(2)+0.77*sin(thetaOcc/2)-0.13,'\\theta_{occ,b}', ...\n    'Color',cOcc,'FontWeight','bold','FontSize',11);",
     "text(ax,observer(1)+1.08*cos(thetaOcc/2), ...\n    observer(2)+1.08*sin(thetaOcc/2)-0.23,'\\theta_{occ,b}', ...\n    'Color',cOcc,'FontWeight','bold','FontSize',13);",
     'occultation label'),
    ("text(ax,observer(1)+1.28*cos(thetaKeepout/2), ...\n    observer(2)+1.28*sin(thetaKeepout/2)+0.07,'\\theta_{keepout,b}', ...\n    'Color',cSensor,'FontWeight','bold','FontSize',11);",
     "text(ax,observer(1)+1.72*cos(thetaKeepout/2), ...\n    observer(2)+1.72*sin(thetaKeepout/2)+0.13,'\\theta_{keepout,b}', ...\n    'Color',cSensor,'FontWeight','bold','FontSize',13);",
     'keepout label'),
    ("text(ax,observer(1)+1.89*cos(thetaTarget/2), ...\n    observer(2)+1.89*sin(thetaTarget/2)+0.08,'\\theta_b', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',11);",
     "text(ax,observer(1)+2.42*cos(thetaTarget/2), ...\n    observer(2)+2.42*sin(thetaTarget/2)+0.18,'\\theta_b', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',13);",
     'target angle label'),
    ("text(ax,-0.70,-0.64,'physical occultation', ...\n    'Color',cOcc,'FontSize',10.2,'FontAngle','italic', ...\n    'HorizontalAlignment','center');\ntext(ax,-0.52,0.55,'additional sensor margin', ...\n    'Color',cSensor,'FontSize',10.2,'FontAngle','italic', ...\n    'HorizontalAlignment','center');",
     "text(ax,-0.42,-1.10,{'physical';'occultation'}, ...\n    'Color',cOcc,'FontSize',12,'FontAngle','italic', ...\n    'FontWeight','bold','HorizontalAlignment','center');\ntext(ax,0.18,1.16,{'sensor';'margin'}, ...\n    'Color',cSensor,'FontSize',12,'FontAngle','italic', ...\n    'FontWeight','bold','HorizontalAlignment','center');",
     'region labels'),
    ("title(ax,'Unified minimum angular-separation geometry', ...\n    'FontSize',13.5,'FontWeight','bold');\nxlim(ax,[-3.05,2.55]);\nylim(ax,[-1.55,3.18]);",
     "xlim(ax,[-3.12,2.78]);\nylim(ax,[-1.95,3.34]);",
     'title and limits'),
    ("'FontName','Times New Roman','FontWeight','bold','FontSize',11.2);",
     "'FontName','Times New Roman','FontWeight','bold','FontSize',13);",
     'inset heading'),
    ("'FontName','Times New Roman','FontAngle','italic','FontSize',9.8);",
     "'FontName','Times New Roman','FontAngle','italic','FontSize',11);",
     'inset note'),
    ("'FontWeight','bold','FontSize',11.7,'Color',cSensor, ...",
     "'FontWeight','bold','FontSize',12.6,'Color',cSensor, ...",
     'footer line one'),
    ("'FontWeight','bold','FontSize',10.8,'VerticalAlignment','middle');",
     "'FontWeight','bold','FontSize',11.6,'VerticalAlignment','middle');",
     'footer line two'),
    ("'FontSize',10.8,'VerticalAlignment','middle');",
     "'FontSize',11.6,'VerticalAlignment','middle');",
     'footer line three')
]

for old, new, name in vis_replacements:
    if old not in vis:
        raise RuntimeError(f'Missing visibility block: {name}')
    vis = vis.replace(old, new, 1)

# Increase the repeated body/value labels in the inset without touching
# other figures.
vis = vis.replace("'FontName','Times New Roman','FontWeight','bold','FontSize',11);",
                  "'FontName','Times New Roman','FontWeight','bold','FontSize',12.5);")
vis = vis.replace("'FontSize',11);", "'FontSize',12.5);")

meas_replacements = [
    ("text(ax1,4.40,-0.08,'x','FontWeight','bold','FontSize',14);",
     "text(ax1,4.40,-0.08,'x','FontWeight','bold','FontSize',16);",
     'x label'),
    ("text(ax1,-0.10,3.62,'y','FontWeight','bold','FontSize',14);",
     "text(ax1,-0.10,3.62,'y','FontWeight','bold','FontSize',16);",
     'y label'),
    ("text(ax1,0,-0.39,'Observer','Color',cObserver, ...\n    'FontWeight','bold','FontSize',12,'HorizontalAlignment','center');",
     "text(ax1,0,-0.43,'Observer','Color',cObserver, ...\n    'FontWeight','bold','FontSize',14,'HorizontalAlignment','center');",
     'left observer'),
    ("text(ax1,projection(1)+0.06,projection(2)+0.22,'LOS projection', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',11);",
     "text(ax1,projection(1)+0.10,projection(2)+0.24,'Target projection', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',13);",
     'projection label'),
    ("text(ax1,1.82,1.28,'\\rho_{xy}', ...\n    'Color',cProjection,'FontWeight','bold','FontSize',12);",
     "text(ax1,1.80,1.78,'\\rho_{xy}', ...\n    'Color',cProjection,'FontWeight','bold','FontSize',14);",
     'left rho xy'),
    ("text(ax1,1.38*cos(alpha/2),1.38*sin(alpha/2)+0.03,'\\alpha', ...\n    'Color',cAngle,'FontWeight','bold','FontSize',15);",
     "text(ax1,1.45*cos(alpha/2),1.45*sin(alpha/2)+0.05,'\\alpha', ...\n    'Color',cAngle,'FontWeight','bold','FontSize',17);",
     'alpha label'),
    ("title(ax1,'(a) Right ascension, \\alpha', ...\n    'FontSize',13.5,'FontWeight','bold');",
     "title(ax1,'(a) Right ascension, \\alpha', ...\n    'FontSize',15,'FontWeight','bold');",
     'left title'),
    ("text(ax2,4.48,-0.08,'\\rho_{xy}', ...\n    'Color',cProjection,'FontWeight','bold','FontSize',12);",
     "text(ax2,4.48,0.18,'\\rho_{xy}', ...\n    'Color',cProjection,'FontWeight','bold','FontSize',14);",
     'right rho xy axis'),
    ("text(ax2,-0.12,3.42,'z','FontWeight','bold','FontSize',14);",
     "text(ax2,-0.12,3.42,'z','FontWeight','bold','FontSize',16);",
     'z label'),
    ("text(ax2,0,-0.39,'Observer','Color',cObserver, ...\n    'FontWeight','bold','FontSize',12,'HorizontalAlignment','center');",
     "text(ax2,0,-0.43,'Observer','Color',cObserver, ...\n    'FontWeight','bold','FontSize',14,'HorizontalAlignment','center');",
     'right observer'),
    ("text(ax2,target(1)+0.08,target(2)+0.20,'Target', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',12);",
     "text(ax2,target(1)+0.10,target(2)+0.22,'Target', ...\n    'Color',cTarget,'FontWeight','bold','FontSize',14);",
     'target label'),
    ("text(ax2,2.15,1.12,'\\rho','FontWeight','bold','FontSize',13);",
     "text(ax2,2.10,1.62,'\\rho','FontWeight','bold','FontSize',15);",
     'rho label'),
    ("text(ax2,1.42*cos(delta/2),1.42*sin(delta/2)+0.03,'\\delta', ...\n    'Color',cAngle,'FontWeight','bold','FontSize',15);",
     "text(ax2,1.48*cos(delta/2),1.48*sin(delta/2)+0.05,'\\delta', ...\n    'Color',cAngle,'FontWeight','bold','FontSize',17);",
     'delta label'),
    ("title(ax2,'(b) Declination, \\delta', ...\n    'FontSize',13.5,'FontWeight','bold');",
     "title(ax2,'(b) Declination, \\delta', ...\n    'FontSize',15,'FontWeight','bold');",
     'right title'),
    ("'FontWeight','bold','FontSize',11.8);",
     "'FontWeight','bold','FontSize',13);",
     'equation footer')
]

for old, new, name in meas_replacements:
    if old not in meas:
        raise RuntimeError(f'Missing measurement block: {name}')
    meas = meas.replace(old, new, 1)

new_text = prefix + vis + meas + suffix

assert "text(ax1,1.80,1.78,'\\rho_{xy}'" in new_text
assert "text(ax2,2.10,1.62,'\\rho'" in new_text
assert "text(ax,observer(1)+2.42*cos(thetaTarget/2)" in new_text

path.write_text(new_text)
print('Applied schematic readability patch v5.')
