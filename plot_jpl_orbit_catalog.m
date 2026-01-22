% ----- plot_jpl_orbit_catalog.m ----- %
close all;
clear;
clc;

% 1. Load Data
if isfile('JPL_CR3BP_OrbitCatalog.mat')
    load('JPL_CR3BP_OrbitCatalog.mat');
else
    error('Data file not found. Please run the data processing script first.');
end

% 2. Define Constants
mu = 1.215058560962404E-2; 

% Family definitions
families = {
    ["NHL1", "NHL2"], ...     
    ["SHL1", "SHL2"], ...     
    ["NNRHL1", "NNRHL2"], ... 
    ["SNRHL1", "SNRHL2"]      
};

% Filenames
filenames = ["northern_halo", "southern_halo", ...
             "northern_rectilinear", "southern_rectilinear"];

% 3. Plotting Loop
for i = 1:4
    fig = figure('Color', 'w', 'WindowState', 'maximized'); 
    hold on; grid on; box on;
    axis equal;
    
    if i <= 2
        view([-35, 30]); % Halo view (Standard)
    else
        view([-50, 25]); % Rectilinear view 
    end
    
    current_pair = families{i};
    colors = ['b', 'r']; 
    h = gobjects(3,1); 
    
    % Plot Both L1 and L2
    for k = 1:2
        targetFamily = current_pair(k);
        idx = T.orbitFamily == targetFamily;
        subT = T(idx, :);
        
        numOrbits = height(subT);
        if numOrbits > 0
            plot_stride = max(1, round(numOrbits/15)); 
            rows_to_plot = 1:plot_stride:numOrbits;
            
            for r = rows_to_plot
                state = subT.state{r};
                p = plot3(state(:,1), state(:,2), state(:,3), ...
                    colors(k), 'LineWidth', 2.0); 
                
                if r == 1
                    h(k) = p; 
                end
            end
        end
    end
    
    % Plot Moon
    h(3) = plot3(1-mu, 0, 0, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 12);
    
    % Formatting (Size 32+) - NO TITLE
    xlabel('X (LU)', 'FontSize', 32, 'FontWeight', 'bold'); 
    ylabel('Y (LU)', 'FontSize', 32, 'FontWeight', 'bold'); 
    zlabel('Z (LU)', 'FontSize', 32, 'FontWeight', 'bold');
    
    % Axis ticks
    ax = gca;
    set(ax, 'FontSize', 32, 'LineWidth', 2.0);
    
    % Legend
    validHandles = isgraphics(h);
    if any(validHandles)
        labels = {'L1', 'L2', 'Moon'};
        lgd = legend(h(validHandles), labels(validHandles), ...
            'Location', 'best'); 
        lgd.FontSize = 28; 
    end
    
    % 4. Save File
    outName = sprintf('%s.eps', filenames(i));
    exportgraphics(fig, outName, 'ContentType', 'vector');
    fprintf('Saved %s\n', outName);
    
    close(fig); 
end