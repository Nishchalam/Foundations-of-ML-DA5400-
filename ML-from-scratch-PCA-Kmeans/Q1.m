%% (1) You are given a dataset with 1000 data points each in R2 . 
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

%% (a) Write a piece of code to run the PCA algorithm on this data-set. How much of  the variance in the data-set is explained by each of the principal components?
dataTable = readtable("dataset1-assignment1 - Sheet1.csv");
data = table2array(dataTable);

% Center the data
avg = compute_average(data);
centered_data = data - avg;

%plotting the data for better visulisation
figure;
hold on;
scatter(data(:,1), data(:,2), 20, "red", "filled",'MarkerFaceAlpha',0.6);
scatter(centered_data(:,1), centered_data(:,2), 20, "blue", "filled",'MarkerFaceAlpha',0.6);
plot(mean(data(:,1)), mean(data(:,2)), "ks", "MarkerSize",10, "MarkerFaceColor","r"); % original mean
plot(mean(centered_data(:,1)), mean(centered_data(:,2)), "gs", "MarkerSize",10, "MarkerFaceColor","b"); % centered mean
title("Data visualization (Original vs Centered)");
xlabel("X-values");
ylabel("Y-values");
legend("Original Data", "Centered Data",'mean of data','mean of centered data');
grid on;
hold off;

%computing covariance for the centered data
cov_mat=compute_covariance(centered_data);

%eigen decomposition of covariance matrix
[eig_vec, eig_val_mat]=eig(cov_mat);
[eig_vals_sorted, indices]=do_sort(diag(eig_val_mat));
%these eigen vectors are the vectors which are in line with most variance
%in the data
eig_vec_sorted=eig_vec(:, indices);

%ratio of explained variance 
evr=eig_vals_sorted/do_sum(eig_vals_sorted);
disp('Ratio by each principal component:');
disp(evr);

%Plotting centered data with PCs overlay
figure;
hold on;
scatter(centered_data(:,1),centered_data(:,2), 20,"green","filled",'MarkerFaceAlpha',0.6);
scale=2; % scaling factor to make arrows visible
%using quiver plot to plot arrows 
quiver(0,0,scale*eig_vec_sorted(1,1),scale*eig_vec_sorted(2,1),'r','LineWidth',2,'MaxHeadSize',2);
quiver(0,0,scale*eig_vec_sorted(1,2),scale*eig_vec_sorted(2,2),'b','LineWidth',2,'MaxHeadSize',2);
title("Centered Data with Principal Component Directions");
xlabel("X-values");
ylabel("Y-values");
legend("Centered Data","PC1","PC2");
grid on;
axis equal;
hold off;

%Explained Variance Ratio Bar Plot
figure;
bar(evr,'FaceColor',[0.2 0.4 0.8]);
title("Explained variance (fraction) per component");
xlabel("Principal Component");
ylabel("Explained Variance Ratio");
grid on;

% Project data into PC space
projected_data=centered_data*eig_vec_sorted;

% Projection of data into PC space (PC1 vs PC2)
figure;
scatter(projected_data(:,1), projected_data(:,2), 20, 'filled');
title("Projection of Centered Data onto Principal Components");
xlabel("PC1");
ylabel("PC2");
grid on;


%Reconstruct from PCs (using all components)
reconstructed_data=projected_data*eig_vec_sorted';

%Compute reconstruction error (Mean Squared Error)
%this helps us analyse if the data is the same even after PCA. If the MSE
%is very low, implies the nonlinearity is retained and PCA was useless
errors=centered_data-reconstructed_data;
[m,~] = size(errors);
row_sums = zeros(m,1);
for i = 1:m
    row_sums(i) = do_sum(errors(i,:).^2);
end
mse =do_mean(row_sums);

% Plot comparison
figure;
hold on;
scatter(centered_data(:,1),centered_data(:,2),20,'b','filled','MarkerFaceAlpha',0.5);
scatter(reconstructed_data(:,1),reconstructed_data(:,2),20,'r','filled','MarkerFaceAlpha',0.5);
title("Centered Data vs PCA Reconstruction");
xlabel("X-values");
ylabel("Y-values");
legend("Original Centered Data","Reconstructed from PCs");
grid on;
axis equal;

% Add error text on the plot
dim = [.15 .8 .1 .1]; % position of annotation (normalized figure units)
annotation('textbox',dim,'String',sprintf('Reconstruction MSE = %.4f',mse),'FitBoxToText','on','BackgroundColor','w');
hold off;

pc1_vector = eig_vec_sorted(:,1);         % First principal component
projected_pc1 = centered_data * pc1_vector; % Projection onto PC1 (1D)
reconstructed_pc1 = projected_pc1 * pc1_vector'; % Back to 2D

% Compute reconstruction error (MSE) for PC1 only
errors_pc1 = centered_data - reconstructed_pc1;
% errors=centered_data-reconstructed_data;
[m,~] = size(errors_pc1);
row_sums = zeros(m,1);
for i = 1:m
    row_sums(i) =do_sum(errors_pc1(i,:).^2);
end
mse_pc1 =do_mean(row_sums);

% Plot comparison: original vs reconstruction using only PC1
figure;
hold on;
scatter(centered_data(:,1), centered_data(:,2), 20, 'b','filled','MarkerFaceAlpha',0.5);
scatter(reconstructed_pc1(:,1), reconstructed_pc1(:,2), 20, 'r','filled','MarkerFaceAlpha',0.5);
title("Centered Data vs PCA Reconstruction (Only PC1)");
xlabel("X-values"); ylabel("Y-values");
legend("Original Centered Data","Reconstruction using PC1");
grid on; axis equal;

% Show error as text
dim = [.15 .8 .1 .1];
annotation('textbox',dim,'String',sprintf('MSE (PC1 only) = %.4f',mse_pc1),...
           'FitBoxToText','on','BackgroundColor','w');
hold off;


%% (b) Write a piece of code to implement the Kernel PCA algorithm on this dataset.Explore various kernels discussed in class.For each Kernel, plot the projection of each point in the dataset onto the top-2 principal components. Use one plot for each kernel - In case of RBF kernel, use a different plot for each value of σ that you use.

%Polynomial kernels
figure;
for d = 2:5
    % Polynomial kernel matrix
    K = (centered_data*centered_data' + 1).^d;
    
    % Eigen decomposition
    [e_vec,e_val_mat] = eig(K);
    [e_val_sorted,indices] = do_sort(diag(e_val_mat));
    e_vec_sorted = e_vec(:,indices);

    % Normalize eigenvectors
    alpha_k=zeros(size(e_vec_sorted));
    for k=1:length(e_val_sorted)
        if e_val_sorted(k) > 1e-8
            alpha_k(:,k) = e_vec_sorted(:,k) / sqrt(e_val_sorted(k));
        end
    end
    
    % Project onto top-2 components
    projected_kpca = K * alpha_k(:,1:2);
    
    % Subplot index (d-1 goes from 1 to 9)
    subplot(2,2,d-1);
    scatter(projected_kpca(:,1),projected_kpca(:,2),10,'filled');
    title(sprintf('Poly deg=%d', d));
    xlabel('PC1'); ylabel('PC2'); grid on;
end
sgtitle('Kernel PCA with Polynomial Kernels (degrees 2–10)');

% RBF kernels subplot
rbf_sigmas = [0.5, 1, 1.5, 2, 3, 4, 5];
figure;
for i = 1:length(rbf_sigmas)
    sigma = rbf_sigmas(i);
    sq_dists = do_pdist2(centered_data,centered_data); 
    K = exp(-sq_dists/(2*sigma^2));
    
    % Eigen decomposition
    [e_vec,e_val_mat] = eig(K);
    [e_val_sorted,indices] = do_sort(diag(e_val_mat));
    e_vec_sorted = e_vec(:,indices);

    % Normalize eigenvectors
    alpha_k = zeros(size(e_vec_sorted));
    for k = 1:length(e_val_sorted)
        if e_val_sorted(k) > 1e-8
            alpha_k(:,k) = e_vec_sorted(:,k) / sqrt(e_val_sorted(k));
        end
    end
    
    % Project onto top-2 components
    projected_kpca = K * alpha_k(:,1:2);
    
    % Subplot index (2x4 grid → room for 8 plots)
    subplot(2,4,i);
    scatter(projected_kpca(:,1),projected_kpca(:,2),10,'filled');
    title(sprintf('\\sigma=%.1f', sigma));
    xlabel('PC1'); ylabel('PC2'); grid on;
end
sgtitle('Kernel PCA with RBF Kernels (σ = 0.5, 1, 1.5, 2, 3, 4, 5)');

