clc; clear; close all;


npts = 100;
P = randn(3, npts); 
P(1,:) = P(1,:) * 3; 
P(2,:) = P(2,:) * 1; 
P(3,:) = P(3,:) * 0.5;


R_rand = eul2rotm([pi/6, pi/4, pi/3]); 

th = pi/4; 
Rx = [1 0 0; 0 cos(th) -sin(th); 0 sin(th) cos(th)];
Ry = [cos(th) 0 sin(th); 0 1 0; -sin(th) 0 cos(th)];
P = Rx * Ry * P;

%% 2.Our-Strategy (Greedy Subset Selection)

% Centroid
Pc = mean(P, 2);

% Pb1: farthest from centroid
[~, idx1] = max(vecnorm(P - Pc, 2, 1));
Pb1 = P(:, idx1);

% Pb2: farthest from Pb1
[~, idx2] = max(vecnorm(P - Pb1, 2, 1));
Pb2 = P(:, idx2);

% Pb3: farthest from line (Pb1, Pb2)
v12 = Pb2 - Pb1;
v12 = v12 / norm(v12);
V12 = repmat(v12, 1, size(P,2));
dist_line = vecnorm(cross(P - Pb1, V12), 2, 1);
[~, idx3] = max(dist_line);
Pb3 = P(:, idx3);

% Pb4: farthest from plane (Pb1, Pb2, Pb3)
n123 = cross(Pb2 - Pb1, Pb3 - Pb1);
n123 = n123 / norm(n123);
dist_plane = abs(n123' * (P - Pb1));
[~, idx4] = max(dist_plane);
Pb4 = P(:, idx4);

bases_our = [Pb1, Pb2, Pb3, Pb4];

%% 3. B: EPnP-Strategy (Virtual Control Points via PCA)

% 1. 
Cw = mean(P, 2);

% 2. 
X_centered = P - Cw;
[U, ~, ~] = svd(X_centered * X_centered');

% 3. 

scale = sqrt(var(P, 0, 2)) * 2; 
if sum(scale)==0, scale = [1;1;1]; end 


C1 = Cw;
C2 = Cw + U(:,1) * scale(1) * 1.5; 
C3 = Cw + U(:,2) * scale(2) * 1.5; 
C4 = Cw + U(:,3) * scale(3) * 1.5; 

bases_epnp = [C1, C2, C3, C4];

%% 4. 

figure('Color','w', 'Position', [100, 100, 1000, 700]); 
hold on; grid on; axis equal;
view(30, 20);


h_pts = scatter3(P(1,:), P(2,:), P(3,:), 20, [0.8 0.8 0.8], 'filled', 'DisplayName', '3D Points');

% ---------------------------------------------------------
% 4.2  Our-Strategy (Real Points Subset)
% ---------------------------------------------------------
%
h_our = scatter3(bases_our(1,:), bases_our(2,:), bases_our(3,:), ...
    120, 'b', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1.5, ...
    'DisplayName', 'Our-Strategy');


faces = [1 2 3; 1 2 4; 1 3 4; 2 3 4];
patch('Vertices', bases_our', 'Faces', faces, ...
      'FaceColor', 'b', 'FaceAlpha', 0.1, ...
      'EdgeColor', 'b', 'LineStyle', '-', 'LineWidth', 1.5);


labels_our = {'P_{b1}', 'P_{b2}', 'P_{b3}', 'P_{b4}'};
for i = 1:4
    text(bases_our(1,i), bases_our(2,i), bases_our(3,i), ['  ' labels_our{i}], ...
         'Color', 'b', 'FontSize', 11, 'FontWeight', 'bold');
end

% ---------------------------------------------------------
% 4.3 EPnP-Strategy (Virtual PCA Points) 
% ---------------------------------------------------------
% 
h_epnp = scatter3(bases_epnp(1,:), bases_epnp(2,:), bases_epnp(3,:), ...
    120, 'r', 'filled', 's', 'MarkerEdgeColor', 'w', 'LineWidth', 1.5, ...
    'DisplayName', 'EPnP-Strategy');


patch('Vertices', bases_epnp', 'Faces', faces, ...
      'FaceColor', 'r', 'FaceAlpha', 0.05, ...
      'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 1.5);


labels_epnp = {'C_{1}^{(PCA)}', 'C_{2}^{(PCA)}', 'C_{3}^{(PCA)}', 'C_{4}^{(PCA)}'};
for i = 1:4
    text(bases_epnp(1,i), bases_epnp(2,i), bases_epnp(3,i), ['  ' labels_epnp{i}], ...
         'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold');
end


for i=2:4
    plot3([bases_epnp(1,1), bases_epnp(1,i)], ...
          [bases_epnp(2,1), bases_epnp(2,i)], ...
          [bases_epnp(3,1), bases_epnp(3,i)], ...
          'r-', 'LineWidth', 2);
end


xlabel('X'); ylabel('Y'); zlabel('Z');
legend([h_pts, h_our, h_epnp], 'Location', 'best', 'FontSize', 12);

set(gca, 'FontSize', 12);
rotate3d on;
