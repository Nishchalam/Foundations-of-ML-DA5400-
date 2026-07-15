%% (2) You are given a data-set with 1000 data points each in R2.
clear all;
close all;

% User defined function to compute mean across columns of a mxn array
function average = compute_average(data)
    [m,n] = size(data);
    average = zeros(1,n); 
    for i=1:n
        col_sum = 0;
        for j=1:m
            col_sum = col_sum + data(j,i);
        end
        average(i) = col_sum / m;
    end
end
%User defined function for computing covariance of data
function covariance_mat = compute_covariance(data)
    [m,n]=size(data);
    avg=compute_average(data);
    centered_data=data-avg;
    covariance_mat=(centered_data'*centered_data)/m;
end
%User defined function for sorting eigenvalues in descending order. Returns both sorted values and the permutation indices
function [sorted_vals, sorted_indices] = do_sort(vals)
    n = length(vals);
    sorted_vals = vals;
    sorted_indices = 1:n;
    
    for i = 1:n-1
        max_idx = i;
        for j = i+1:n
            if sorted_vals(j) > sorted_vals(max_idx)
                max_idx = j;
            end
        end
        % Swap values
        temp = sorted_vals(i);
        sorted_vals(i) = sorted_vals(max_idx);
        sorted_vals(max_idx) = temp;
        % Swap indices
        temp_idx = sorted_indices(i);
        sorted_indices(i) = sorted_indices(max_idx);
        sorted_indices(max_idx) = temp_idx;
    end
end
% User defined function to compute sum of elements in a vector
function total = do_sum(vec)
    total = 0;
    for i = 1:length(vec)
        total = total + vec(i);
    end
end
% User defined function to compute mean of a vector
function mu = do_mean(vec)
    n = length(vec);
    total = 0;
    for i = 1:n
        total = total + vec(i);
    end
    mu = total / n;
end
% User defined function to compute squared Euclidean distance matrix
function D = do_pdist2(A, B)
    [m, d1] = size(A);
    [n, d2] = size(B);
    if d1 ~= d2
        error('Input matrices must have the same number of columns');
    end
    D = zeros(m, n);
    for i = 1:m
        for j = 1:n
            diff = A(i,:) - B(j,:);
            D(i,j) = sum(diff.^2);
        end
    end
end
%user-defined function for randomising 
function idx = do_randperm(n, k)
    if nargin < 2
        k = n;
    end
    idx = 1:n;
    for i = n:-1:2
        j = floor(rand() * i) + 1; 
        temp = idx(i);
        idx(i) = idx(j);
        idx(j) = temp;
    end
    % truncate if needed
    idx = idx(1:k);
end

%% (a) Write a piece of code to run the algorithm studied in class for the K-means
% problem with k = 4. Try 5 different random initialization and plot the error
% function w.r.t. iterations in each case. In each case, plot the clusters obtained
% in different colors.
dataTable = readtable("Dataset2-Assignment 1 - Dataset.csv");
data = table2array(dataTable);

% center the data
avg = compute_average(data);
centered_data = data - avg;

%plotting the data for better visulisation
figure;
hold on;
scatter(centered_data(:,1), centered_data(:,2), 20, "blue", "filled",'MarkerFaceAlpha',0.6);
plot(mean(centered_data(:,1)), mean(centered_data(:,2)), "gs", "MarkerSize",10, "MarkerFaceColor","b"); % centered mean
title("Data visualization (Original vs Centered)");
xlabel("X-values");
ylabel("Y-values");
legend("Centered Data",'mean of centered data');
grid on;
hold off;

function [centers, labels, J_hist] = my_kmeans(X, K, max_iter, tol)
    [m, d] = size(X);
    rand_idx = do_randperm(m, K); 
    centers = X(rand_idx, :);
    J_hist = zeros(max_iter, 1);
    labels = zeros(m, 1);
    for it = 1:max_iter
        D = zeros(m, K); 
        for k = 1:K
            diff = X - centers(k, :); 
            for i = 1:m
                D(i,k) = do_sum(diff(i,:).^2); 
            end
        end
        [minDist, labels] = min(D, [], 2);
        new_centers = zeros(K, d);
        for k = 1:K
            idx = find(labels == k);
            if isempty(idx)
                new_centers(k,:) = X(randi(m), :);
            else
                for j = 1:d
                    new_centers(k,j) = do_mean(X(idx,j));
                end
            end
        end
        centers = new_centers;
        J_hist(it) = do_sum(minDist);
        if it > 1 && abs(J_hist(it) - J_hist(it-1)) < tol
            J_hist = J_hist(1:it);
            break;
        end
    end
end

K = 4;
max_iter = 100;
tol = 1e-6;
n_restarts = 5;

all_J = cell(n_restarts,1);
all_labels = cell(n_restarts,1);
all_centers = cell(n_restarts,1);

figure;
hold on;
colors = lines(n_restarts);

for r = 1:n_restarts
    [centers, labels, J_hist] = my_kmeans(centered_data, K, max_iter, tol);
    all_J{r} = J_hist;
    all_labels{r} = labels;
    all_centers{r} = centers;

    % Plot error vs iterations
    plot(1:length(J_hist), J_hist, '-o', 'Color', colors(r,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('Run %d', r));
end

xlabel('Iterations');
ylabel('Objective J');
title('K-means Error vs Iterations (5 runs)');
legend show;
grid on;
hold off;

figure;
for r = 1:n_restarts
    subplot(2,3,r); % 2x3 grid for 5 runs
    gscatter(centered_data(:,1), centered_data(:,2), all_labels{r});
    hold on;
    plot(all_centers{r}(:,1), all_centers{r}(:,2), 'kx', ...
        'MarkerSize', 12, 'LineWidth', 2);
    title(sprintf('K-means clustering (Run %d)', r));
    xlabel('X1'); ylabel('X2');
    grid on; axis equal;
end


%% (b) Fix a random initialization. For K = {2, 3, 4, 5}, obtain cluster centers according
% to K-means algorithm using the fixed initialization. For each value of K, plot the
% Voronoi regions associated to each cluster center. (You can assume the minimum
% and maximum value in the data-set to be the range for each component of R2 ).
K_values = [2, 3, 4, 5];
max_iter = 100;
tol = 1e-6;

% Compute global axis limits based on dataset
x_min = min(centered_data(:,1));
x_max = max(centered_data(:,1));
y_min = min(centered_data(:,2));
y_max = max(centered_data(:,2));

% Expand limits slightly for nicer display
margin = 2;
x_limits = [x_min-margin, x_max+margin];
y_limits = [y_min-margin, y_max+margin];

figure;
for idx = 1:length(K_values)
    K = K_values(idx);

    % Run K-means (fixed initialization)
    rng(1); % reproducibility
    [centers, labels, J_hist] = my_kmeans(centered_data, K, max_iter, tol);

    % Subplot
    subplot(2,2,idx);
    hold on;

    % Plot clusters
    colors = lines(K);
    for k = 1:K
        scatter(centered_data(labels==k,1), centered_data(labels==k,2), ...
            15, colors(k,:), 'filled');
    end

    % Plot cluster centers
    scatter(centers(:,1), centers(:,2), 100, 'kx', 'LineWidth', 2);

    % Voronoi regions
    if K >= 3
        [vx, vy] = voronoi(centers(:,1), centers(:,2));
        plot(vx, vy, 'k--', 'LineWidth', 1);
    elseif K == 2
        % Perpendicular bisector
        c1 = centers(1,:);
        c2 = centers(2,:);
        mid = (c1 + c2) / 2;
        dir = c2 - c1;
        perp = [-dir(2), dir(1)];
        t = linspace(-50,50,200);
        bisector = mid + t' * perp;
        plot(bisector(:,1), bisector(:,2), 'k--', 'LineWidth', 1);
    end

    % Fix axis limits and equal aspect ratio
    xlim(x_limits);
    ylim(y_limits);
    axis equal; % equal scaling

    title(sprintf('K-means Clustering with K=%d', K));
    xlabel('X1'); ylabel('X2');
    grid on; hold off;
end
sgtitle('K-means Clustering with Voronoi Regions (K=2,3,4,5)');



%% (c) Run the spectral clustering algorithm (spectral relaxation of K-means using
% Kernel- PCA) k = 4. Choose an appropriate kernel for this data-set and plot
% the clusters obtained in different colors. Explain your choice of kernel based on
% the output you obtain.


% Step 1: Build RBF Kernel
sigma = 2;
sq_dists = do_pdist2(centered_data, centered_data); % user-defined function
K = exp(-sq_dists / (2*sigma^2));

% Step 2: Eigen decomposition
[e_vec, e_val_mat] = eig(K);
[e_val_sorted, indices] = do_sort(diag(e_val_mat)); % user-defined sort
e_vec_sorted = e_vec(:, indices);

% Step 3: Select top-k eigenvectors
k = 4;
U = e_vec_sorted(:, 1:k);

% Normalize rows of U
U_norm = U ./ sqrt(sum(U.^2, 2));

% Step 4: K-means on U_norm
max_iter = 100;
tol = 1e-6;
[centers_spec, labels_spec, ~] = my_kmeans(U_norm, k, max_iter, tol);

% Step 5: Plot clusters in original space
figure;
hold on;
colors = lines(k);
for cluster = 1:k
    scatter(centered_data(labels_spec==cluster,1),centered_data(labels_spec==cluster,2),20, colors(cluster,:), 'filled');
end
title(sprintf('Spectral Clustering (RBF Kernel, σ=%.1f, k=%d)', sigma, k));
xlabel('X1'); ylabel('X2');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4');
grid on; hold off;

%% (c) Spectral Clustering with Polynomial Kernel
% Parameters
k = 4;             % number of clusters
poly_degree = 3;   % polynomial degree
c = 1;             % constant term

% Step 1: Build Polynomial Kernel
K = (centered_data * centered_data' + c).^poly_degree;

% Step 2: Eigen decomposition
[e_vec, e_val_mat] = eig(K);
[e_val_sorted, indices] = do_sort(diag(e_val_mat)); % user-defined sort
e_vec_sorted = e_vec(:, indices);

% Step 3: Select top-k eigenvectors
U = e_vec_sorted(:, 1:k);

% Step 4: Normalize rows of U
U_norm = U ./ sqrt(sum(U.^2, 2));

% Step 5: K-means on U_norm
max_iter = 100;
tol = 1e-6;
[centers_poly, labels_poly, ~] = my_kmeans(U_norm, k, max_iter, tol);

% Step 6: Plot clusters in original space
figure;
hold on;
colors = lines(k);
for cluster = 1:k
    scatter(centered_data(labels_poly==cluster,1), centered_data(labels_poly==cluster,2), 20, colors(cluster,:), 'filled');
end
title(sprintf('Spectral Clustering (Polynomial Kernel, deg=%d, k=%d)', poly_degree, k));
xlabel('X1'); ylabel('X2');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4');
grid on; hold off;

%% (d) Instead of using the method suggested by spectral clustering to map eigenvectors
% to cluster assignments, use the following method: Assign data point i to cluster
% ℓ whenever
% ℓ = arg max vji
% j=1,...,k
% where vj ∈ Rn
%  is the eigenvector of the Kernel matrix associated with the j-th
% largest eigenvalue. How does this mapping perform for this dataset? Explain
% your insights.

% Step 1: Build RBF affinity matrix (same as part c)
sigma = 2.0;
sq_dists = do_pdist2(centered_data, centered_data);
A = exp(-sq_dists/(2*sigma^2));
A(1:size(A,1)+1:end) = 0;
A = (A + A')/2;

% Step 2: Normalized Laplacian
deg = sum(A,2);
DinvSqrt = diag(1 ./ sqrt(deg + eps));
Lsym = eye(size(A)) - DinvSqrt * A * DinvSqrt;

% Step 3: Eigen decomposition
[V,E] = eig(Lsym);
[eigvals_sorted, idx] = sort(diag(E),'ascend');
U = V(:, idx(1:4));   % pick first 4 eigenvectors

% Step 4: Direct mapping (argmax rule)
[~, labels_d] = max(U, [], 2);

% Step 5: Plot results
figure;
gscatter(centered_data(:,1), centered_data(:,2), labels_d);
title('Spectral Clustering - Direct Argmax Mapping (\sigma=2, k=4)');
xlabel('X1'); ylabel('X2'); axis equal; grid on;