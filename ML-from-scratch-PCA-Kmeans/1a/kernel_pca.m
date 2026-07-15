%% Example: Linear PCA vs Kernel PCA on concentric circles
clear; close all; clc;

% Generate concentric circles dataset
n = 1000;
theta = linspace(0,2*pi,n/2)';
circle1 = [cos(theta), sin(theta)] + 0.1*randn(n/2,2);
circle2 = 2*[cos(theta), sin(theta)] + 0.1*randn(n/2,2);
X = [circle1; circle2];

% Plot original data
figure;
subplot(131);
scatter(X(:,1), X(:,2), 20, 'filled');
title('Original Data: Concentric Circles');
xlabel('x1'); ylabel('x2'); axis equal; grid on;
ylim([-2.5 2.5]);

% ---- Linear PCA ----
Xc = X - mean(X);
[coeff,~,~] = pca(Xc); % using MATLAB’s built-in PCA for brevity
Z_linear = Xc * coeff(:,1:2);

subplot(132);
scatter(Z_linear(:,1), Z_linear(:,2), 20, 'filled');
title('Linear PCA Projection');
xlabel('PC1'); ylabel('PC2'); grid on;
ylim([-2.5 2.5]);

% ---- Kernel PCA with RBF ----
sigma = 1;
sq_dists = pdist2(Xc,Xc,'euclidean').^2;
K = exp(-sq_dists/(2*sigma^2));

% Eigen decomposition
[e_vec,e_val] = eig(K);
[e_val_sorted,idx] = sort(diag(e_val),'descend');
e_vec_sorted = e_vec(:,idx);

% Normalize
alpha = e_vec_sorted(:,1:2);
for k = 1:2
    alpha(:,k) = alpha(:,k) / sqrt(e_val_sorted(k));
end

% Projection
Z_kpca = K*alpha;

subplot(133);
scatter(Z_kpca(:,1), Z_kpca(:,2), 20, 'filled');
title(sprintf('Kernel PCA with RBF Kernel (σ=%.1f)', sigma));
xlabel('PC1 (feature space)'); ylabel('PC2 (feature space)'); grid on;
ylim([-2.5 2.5]);
