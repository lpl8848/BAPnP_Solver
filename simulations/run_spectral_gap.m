function run_spectral_gap
% Compute spectral gap for BAPnP, EPnP, CPnP, RPnP, and oDLT
% under quasi-planar configurations.

addpath('D:/R2020b/bin/pnp');
addpath('D:/R2020b/bin/pnp/oDLT/utils');
addpath('D:/R2020b/bin/pnp/oDLT/PnP/oDLT');

n_points = 20; n_trials = 500;
gammas = logspace(-1, -12, 10);

gap_bapnp = zeros(length(gammas), 1);
gap_epnp  = zeros(length(gammas), 1);
gap_cpnp  = zeros(length(gammas), 1);
gap_rpnp  = zeros(length(gammas), 1);
gap_odlt  = zeros(length(gammas), 1);

fprintf('Spectral Gap Analysis (BAPnP vs EPnP vs CPnP vs RPnP vs oDLT)\n');
fprintf('%-10s %-10s %-10s %-10s %-10s %-10s\n', 'Gamma','BAPnP','EPnP','CPnP','RPnP','oDLT');

for i = 1:length(gammas)
    gamma = gammas(i);
    g_b=[]; g_e=[]; g_c=[]; g_r=[]; g_s=[]; g_o=[];

    for k = 1:n_trials
        % Scene generation (same as planar experiment)
        Pw = (rand(3, n_points) - 0.5) * 10;
        Pw(3, :) = Pw(3, :) * gamma;
        R_gt = random_rotation();
        T_gt = [0; 0; 20];
        Pc = R_gt * Pw + T_gt;
        y_2d = Pc(1:2, :) ./ Pc(3, :);

        % Normalization
        cent = mean(Pw, 2);
        P_centered = Pw - cent;
        scale = sqrt(3) / mean(sqrt(sum(P_centered.^2, 1)));
        Pw_norm = P_centered * scale;

        y_homo = [y_2d; ones(1,n_points)];
        y_sphere = y_homo ./ sqrt(sum(y_homo.^2, 1));

        % 1. BAPnP (sigma_3/sigma_1 of 3(n-4)x4 matrix)
        try
            L = get_bapnp_matrix(y_sphere, Pw_norm);
            s = svd(L);
            if length(s) >= 4, g_b = [g_b; s(3)/s(1)]; end
        catch, end

        % 2. EPnP (sigma_11/sigma_1 of 2n x 12)
        try
            M = get_epnp_matrix(y_homo, Pw_norm);
            s = svd(M);
            if length(s) >= 12, g_e = [g_e; s(11)/s(1)]; end
        catch, end

        % 3. CPnP (sigma_11/sigma_1 of 2n x 11)
        try
            Pesi = get_cpnp_matrix(y_2d, Pw_norm);
            s = svd(Pesi);
            if length(s) >= 11, g_c = [g_c; s(11)/s(1)]; end
        catch, end

        % 4. RPnP (sigma_5/sigma_1 of 2n x 6)
        try
            D_rpnp = get_rpnp_matrix(y_2d, Pw_norm, R_gt);
            s = svd(D_rpnp);
            if length(s) >= 6, g_r = [g_r; s(5)/s(1)]; end
        catch, end

        % 5. oDLT (sigma_12/sigma_1 of weighted 2n x 12 DLT matrix)
        try
            A = get_odlt_matrix(y_2d, Pw_norm);
            s = svd(A, 'econ');
            if length(s) >= 12, g_o = [g_o; s(11)/s(1)]; end
        catch, end
    end

    gap_bapnp(i) = median(g_b);
    gap_epnp(i)  = median(g_e);
    gap_cpnp(i)  = median(g_c);
    gap_rpnp(i)  = median(g_r);
    gap_odlt(i)  = median(g_o);

    fprintf('%.1e    %.4e   %.4e   %.4e   %.4e   %.4e\n', ...
        gamma, gap_bapnp(i), gap_epnp(i), gap_cpnp(i), gap_rpnp(i), gap_odlt(i));
end

% Save data
save('D:/R2020b/bin/pnp/spectral_gap_results.mat', ...
    'gammas', 'gap_bapnp', 'gap_epnp', 'gap_cpnp', 'gap_rpnp', 'gap_odlt');

% Plot
figure('Color','w','Position',[300,300,800,450]);
loglog(gammas, gap_bapnp, '-o', 'Color', [0.85,0.33,0.1], 'LineWidth', 2.5, ...
    'MarkerFaceColor', [0.85,0.33,0.1], 'DisplayName', 'BAPnP'); hold on;
loglog(gammas, gap_epnp, '--s', 'Color', [0,0.45,0.74], 'LineWidth', 2, ...
    'MarkerFaceColor', [0,0.45,0.74], 'DisplayName', 'EPnP');
loglog(gammas, gap_cpnp, '-.d', 'Color', [0.47,0.67,0.19], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.47,0.67,0.19], 'DisplayName', 'CPnP');
loglog(gammas, gap_rpnp, '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.49,0.18,0.56], 'DisplayName', 'RPnP');
loglog(gammas, gap_odlt, '-h', 'Color', [0.09,0.75,0.81], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.09,0.75,0.81], 'DisplayName', 'oDLT');

set(gca, 'XDir', 'reverse');
xlabel('Degree of Coplanarity (\gamma)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Spectral Gap (\sigma_{min}/\sigma_{max})', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 10);
ylim([1e-6, 1.5]); grid on;

% Save figure
outdir = 'D:\研究生学习\pnp\pnp\论文写作 - 副本\final';
saveas(gcf, fullfile(outdir, 'spectral_gap.fig'));
exportgraphics(gcf, fullfile(outdir, 'spectral_gap.pdf'), 'ContentType', 'vector');
fprintf('Spectral gap figure saved.\n');
end

%% ========================================================================
%  oDLT Matrix Construction
% ========================================================================
function A = get_odlt_matrix(y_2d, X)
% Build the weighted DLT matrix that oDLT solves via SVD.
% X: 3xN world points, y_2d: 2xN normalized image coords
    N = size(X, 2);
    X_h = [X; ones(1, N)];
    U_h = [y_2d; ones(1, N)];

    % Normalize (same as oDLT's internal normalization)
    X_mean = mean(X, 2);
    X_s = sqrt(3) / mean(sqrt(sum((X - X_mean).^2, 1)));
    X_scaled = ((X - X_mean) * X_s)';

    U_mean = mean(U_h(1:2,:), 2);
    U_s = sqrt(3) / mean(sqrt(sum((U_h(1:2,:) - U_mean).^2, 1)));
    U_scaled = ((U_h(1:2,:) - U_mean) * U_s)';
    U_scaled(:,3) = 1;

    % Build DLT system (same as build_dlt_system in oDLT)
    A = zeros(2*N, 12);
    for i = 1:N
        X_i = [X_scaled(i,:), 1];
        u = U_scaled(i,1); v = U_scaled(i,2);
        A(2*i-1, :) = [X_i, zeros(1,4), -u*X_i];
        A(2*i, :)   = [zeros(1,4), X_i, -v*X_i];
    end

    % Compute weights (oDLT's weighting)
    [~,~,V] = svd(A(1:min(2*N, min(max(40, floor(N/10)), 100)), :), 'econ');
    h = V(:,end);
    H_init = reshape(h, 3, 4);
    scale_3 = H_init(3,:) * [X_scaled, ones(N,1)]';
    q = 1 ./ (1.0 * abs(scale_3));

    % Apply weights
    A = kron(q', [1; 1]) .* A;
end

function L = get_bapnp_matrix(y_norm, P_n)
    N = size(P_n, 2);
    d2 = sum(P_n.^2, 1); [~, b1] = max(d2);
    d2 = sum((P_n - P_n(:,b1)).^2, 1); [~, b2] = max(d2);
    v12 = P_n(:,b2) - P_n(:,b1); v12 = v12/(norm(v12)+eps);
    cp = cross(repmat(v12,1,N), P_n - P_n(:,b1)); [~, b3] = max(sum(cp.^2,1));
    v13 = P_n(:,b3) - P_n(:,b1); n_vec = cross(v12,v13);
    if norm(n_vec)<1e-8, b4=setdiff(1:N,[b1,b2,b3]); b4=b4(1);
    else, n_vec=n_vec/norm(n_vec); [~,b4]=max(abs(n_vec'*(P_n-P_n(:,b1)))); end
    base = [b1,b2,b3,b4]; perm = [base, setdiff(1:N,base)];
    Ps = P_n(:,perm); ys = y_norm(:,perm);
    Basis = Ps(:,1:3)-Ps(:,4); Target = Ps(:,5:end)-Ps(:,4);
    coeffs = pinv(Basis)*Target;
    al=coeffs(1,:); be=coeffs(2,:); ga=coeffs(3,:); de=1-sum(coeffs,1);
    L = zeros(3*(N-4),4); y1=ys(:,1); y2=ys(:,2); y3=ys(:,3); y4=ys(:,4);
    for j=1:(N-4)
        yx=[0 -ys(3,j+4) ys(2,j+4); ys(3,j+4) 0 -ys(1,j+4); -ys(2,j+4) ys(1,j+4) 0];
        r=(j-1)*3+1:j*3; L(r,1)=al(j)*(yx*y1); L(r,2)=be(j)*(yx*y2); L(r,3)=ga(j)*(yx*y3); L(r,4)=de(j)*(yx*y4);
    end
end

function M = get_epnp_matrix(y_homo, P_c)
    N=size(P_c,2); C=P_c*P_c'; [U,S,~]=svd(C);
    si=sqrt(diag(S)); Cw=[zeros(3,1),si(1)*U(:,1),si(2)*U(:,2),si(3)*U(:,3)];
    Ch=[Cw;ones(1,4)]; Ph=[P_c;ones(1,N)]; al=pinv(Ch)*Ph;
    M=zeros(2*N,12); uv=y_homo(1:2,:)./y_homo(3,:);
    for i=1:N
        u=uv(1,i); v=uv(2,i); a=al(:,i);
        for j=1:4
            c=(j-1)*3+1; M(2*i-1,c)=a(j); M(2*i-1,c+2)=-a(j)*u;
            M(2*i,c+1)=a(j); M(2*i,c+2)=-a(j)*v;
        end
    end
end

function P = get_cpnp_matrix(Psens_2D, s)
    N=size(s,2); bar_s=sum(s,2)/N; Psens_2D=Psens_2D;
    obs=Psens_2D(:); P=zeros(2*N,11);
    for k=1:N
        sk=s(:,k); uk=obs(2*k-1); vk=obs(2*k); ds=sk-bar_s;
        P(2*k-1,:)=[-ds(1)*uk,-ds(2)*uk,-ds(3)*uk,sk(1),sk(2),sk(3),1,0,0,0,0];
        P(2*k,:)=[-ds(1)*vk,-ds(2)*vk,-ds(3)*vk,0,0,0,0,sk(1),sk(2),sk(3),1];
    end
end

function D = get_rpnp_matrix(xx, XX, R_gt)
    n=size(xx,2); xxv=[xx;ones(1,n)];
    for i=1:n, xxv(:,i)=xxv(:,i)/norm(xxv(:,i)); end
    i1=1;i2=2; lmin=1.0;
    rij=ceil(rand(min(n,50),2)*n);
    for k=1:size(rij,1)
        i=rij(k,1);j=rij(k,2); if i==j, continue; end
        l=dot(xxv(:,i),xxv(:,j)); if l<lmin, lmin=l;i1=i;i2=j; end
    end
    p1=XX(:,i1);p2=XX(:,i2);p0=(p1+p2)/2;
    x_axis=p2-p0; x_axis=x_axis/norm(x_axis);
    if abs([0 1 0]*x_axis)<abs([0 0 1]*x_axis)
        z_axis=cross(x_axis,[0;1;0]);z_axis=z_axis/norm(z_axis);
        y_axis=cross(z_axis,x_axis);y_axis=y_axis/norm(y_axis);
    else
        y_axis=cross([0;0;1],x_axis);y_axis=y_axis/norm(y_axis);
        z_axis=cross(x_axis,y_axis);z_axis=z_axis/norm(z_axis);
    end
    Ro=[x_axis,y_axis,z_axis]; XXl=Ro'*(XX-repmat(p0,1,n));
    Rx=R_gt*Ro; r=Rx'; D=zeros(2*n,6);
    for j=1:n
        ui=xx(1,j);vi=xx(2,j);xi=XXl(1,j);yi=XXl(2,j);zi=XXl(3,j);
        D(2*j-1,:)=[-r(2)*yi+ui*(r(8)*yi+r(9)*zi)-r(3)*zi, -r(3)*yi+ui*(r(9)*yi-r(8)*zi)+r(2)*zi, -1,0,ui, ui*r(7)*xi-r(1)*xi];
        D(2*j,:)=[-r(5)*yi+vi*(r(8)*yi+r(9)*zi)-r(6)*zi, -r(6)*yi+vi*(r(9)*yi-r(8)*zi)+r(5)*zi, 0,-1,vi, vi*r(7)*xi-r(4)*xi];
    end
end

function R = random_rotation()
    [Q,~]=qr(randn(3)); R=Q;
end
