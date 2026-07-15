%% EE5110 Mini-Quiz: EM for Mixture Models
clear; clc; close all;

%% --- Configuration ---
K = 4;
num_initializations = 100;
max_iters = 100;
tol = 1e-6;
file_name = 'A2Q1.csv';

%% --- Load Data ---
try
    X = readmatrix(file_name);
    [N, D] = size(X);
    fprintf('Data loaded: N=%d data points, D=%d features.\n', N, D);
catch
    error('Could not load data. Ensure A2Q1.csv is in the current directory.');
end

%% --- Q1(i): EM for Mixture of Bernoulli Distributions ---
% Determine which probabilisitic mixture could have generated this data (It is
% not a Gaussian mixture). Derive the EM algorithm for your choice of mixture
% and show your calculations. Write a piece of code to implement the algorithm
% you derived by setting the number of mixtures K = 4. Plot the log-likelihood
% (averaged over 100 random initializations) as a function of iterations.
fprintf('\nRunning EM for Mixture of Bernoulli (K=%d)...\n', K);
log_likelihoods_all = NaN(num_initializations, max_iters);
final_models_bernoulli = cell(num_initializations,1);

for init = 1:num_initializations
    [pi_k, mu_k, LL_hist] = em_bernoulli(X, K, max_iters);
    log_likelihoods_all(init, 1:length(LL_hist)) = LL_hist;
    final_models_bernoulli{init} = struct('pi_k',pi_k,'mu_k',mu_k);
end

avg_LL_bernoulli = nanmean(log_likelihoods_all, 1);

figure;
plot(1:max_iters, avg_LL_bernoulli, 'LineWidth', 2);
xlabel('Iterations'); ylabel('Averaged Log-Likelihood');
title(sprintf('Q1(i): EM for Mixture of Bernoulli (K=%d)', K)); grid on;

% % Display Bernoulli mixture info for last run
% bm = final_models_bernoulli{end};
% fprintf('\n=== Bernoulli Mixture Cluster Parameters ===\n');
% for k = 1:K
%     fprintf('Cluster %d: weight = %.3f\n', k, bm.pi_k(k));
%     fprintf('  mu_k (first 10 dims): '); disp(bm.mu_k(k, 1:min(10,D)));
% end


%% --- Q1(ii): EM for Mixture of Gaussians ---
% (ii) Assume that the same data was infact generated from a mixture of Gaussians
% with 4 mixtures. Implement the EM algorithm and plot the log-likelihood (aver-
% aged over 100 random initializations of the parameters) as a function of iterations.
% How does the plot compare with the plot from part (i)? Provide insights that
% you draw from this experiment.

fprintf('\nRunning EM for Mixture of Gaussians (K=%d)...\n', K);
log_likelihoods_all_gmm = NaN(num_initializations, max_iters);
final_models_gmm = cell(num_initializations,1);

for init = 1:num_initializations
    [pi_k_g, mu_k_g, Sigma_k_g, LL_hist_gmm] = em_gaussian(X, K, max_iters);
    log_likelihoods_all_gmm(init, 1:length(LL_hist_gmm)) = LL_hist_gmm;
    final_models_gmm{init} = struct('pi_k',pi_k_g,'mu_k',mu_k_g,'Sigma_k',Sigma_k_g);
end

avg_LL_gmm = nanmean(log_likelihoods_all_gmm, 1);

figure;
hold on;
plot(1:max_iters, avg_LL_bernoulli, 'LineWidth', 2, 'DisplayName','Bernoulli Mixture');
plot(1:max_iters, avg_LL_gmm, 'LineWidth', 2, 'DisplayName','Gaussian Mixture');
hold off; legend; grid on;
xlabel('Iterations'); ylabel('Averaged Log-Likelihood');
title(sprintf('Q1(ii): Bernoulli vs Gaussian (K=%d, %d runs)', K, num_initializations));

% gm = final_models_gmm{end};
% fprintf('\n=== Gaussian Mixture Cluster Parameters ===\n');
% for k = 1:K
%     fprintf('Cluster %d: weight = %.3f\n', k, gm.pi_k(k));
%     fprintf('  mu_k (first 5 dims): '); disp(gm.mu_k(k,1:min(5,D)));
% end

%% --- Q1(iii): K-means Clustering ---
% Run the K-means algorithm with K = 4 on the same data. Plot the objective of
% K − means as a function of iterations.

fprintf('\nRunning K-Means (K=%d)...\n', K);
objective_histories_kmeans = NaN(num_initializations, max_iters);
centers_all = cell(num_initializations,1);
labels_all = cell(num_initializations,1);

for init = 1:num_initializations
    [centers, labels, J_hist] = my_kmeans(X, K, max_iters, tol);
    objective_histories_kmeans(init,1:length(J_hist)) = J_hist;
    centers_all{init} = centers;
    labels_all{init} = labels;
end

avg_obj = mean(objective_histories_kmeans, 1, 'omitnan');
last_valid_iter = find(~isnan(avg_obj), 1, 'last');
if isempty(last_valid_iter), last_valid_iter = max_iters; end

figure;
plot(1:last_valid_iter, avg_obj(1:last_valid_iter),'r','LineWidth',2);
xlabel('Iterations'); ylabel('Averaged Objective (Sum of Squares)');
title(sprintf('Q1(iii): K-Means Objective (K=%d, %d runs)', K, num_initializations)); grid on;

% fprintf('\n=== K-Means Cluster Centers (last run) ===\n');
% disp(centers_all{end});

% %% --- Visualization for Clusters (optional 2D data only) ---
% if D >= 2
%     labels = labels_all{end};
%     figure;
%     gscatter(X(:,1), X(:,2), labels); hold on;
%     plot(centers_all{end}(:,1), centers_all{end}(:,2), 'kx','MarkerSize',12,'LineWidth',2);
%     title('K-Means Cluster Assignments'); xlabel('Feature 1'); ylabel('Feature 2');
%     hold off;
% end

%% User defined functions
% function for EM algorithm for bernoulli distribution
function [pi_k, mu_k, log_likelihood_history] = em_bernoulli(X, K, max_iters)
[N, D] = size(X);
epsilon = 1e-6;
log_likelihood_history = zeros(1, max_iters);

pi_k = rand(1, K);
pi_k = pi_k/sum(pi_k);
mu_k = epsilon + (1 - 2*epsilon)*rand(K, D);

for iter = 1:max_iters
    log_p_xk = zeros(N, K);
    for k = 1:K
        log_p_xk(:,k) = X*log(mu_k(k,:)') + (1-X)*log(1-mu_k(k,:)');
    end

    log_pi = log(pi_k);
    log_num = log_p_xk + log_pi;          
    m = max(log_num, [], 2);
    log_den = m + log(sum(exp(log_num - m), 2));
    r_nk = exp(log_num - log_den); 

    N_k = sum(r_nk,1);
    pi_k = N_k / N;
    sum_r_x = r_nk' * X;
    mu_k = sum_r_x ./ N_k'; 
    mu_k = min(max(mu_k, epsilon), 1-epsilon);

    log_likelihood_history(iter) = sum(log_den);
    if iter>1 && abs(log_likelihood_history(iter)-log_likelihood_history(iter-1))<1e-5
        log_likelihood_history = log_likelihood_history(1:iter);
        break;
    end
end
end

%function for EM algorithm for gaussian distribution
function [pi_k, mu_k, Sigma_k, log_likelihood_history] = em_gaussian(X, K, max_iters)
[N, D] = size(X);
reg = 1e-6;
log_likelihood_history = zeros(1, max_iters);

pi_k = rand(1,K); 
pi_k = pi_k / sum(pi_k);
mu_k = X(randperm(N,K),:);
Sigma_k = repmat(eye(D),1,1,K);

for iter = 1:max_iters
    log_p_xk = zeros(N,K);
    for k = 1:K
        S = Sigma_k(:,:,k) + eye(D)*reg;
        [R,p] = chol(S);
        if p > 0
            S = S + eye(D)*reg;
            R = chol(S);
        end
        log_det = 2*sum(log(diag(R)));
        Z = (X - mu_k(k,:)) / R;
        log_p_xk(:,k) = -0.5 * (D*log(2*pi) + log_det + sum(Z.^2,2));
    end

    log_num = log_p_xk + log(pi_k);                
    m = max(log_num, [], 2);
    log_den = m + log(sum(exp(log_num - m), 2));
    r_nk = exp(log_num - log_den);                

    N_k = sum(r_nk,1);
    pi_k = N_k / N;
    mu_k = (r_nk' * X) ./ N_k';                    

    for k = 1:K
        Xc = X - mu_k(k,:);
        Xw = Xc .* sqrt(r_nk(:,k));               
        Sigma_k(:,:,k) = (Xw' * Xw) / N_k(k) + eye(D)*reg;
    end

    log_likelihood_history(iter) = sum(log_den);
    if iter > 1 && abs(log_likelihood_history(iter) - log_likelihood_history(iter-1)) < 1e-5
        log_likelihood_history = log_likelihood_history(1:iter);
        break;
    end
end
end

%function to print k-means
function [centers, labels, J_hist] = my_kmeans(X, K, max_iter, tol)
[m,d] = size(X);
idx = randperm(m,K);
centers = X(idx,:);
J_hist = zeros(max_iter,1);
for it = 1:max_iter
    D = pdist2(X, centers).^2;
    [minD, labels] = min(D,[],2);
    for k = 1:K
        if any(labels==k)
            centers(k,:) = mean(X(labels==k,:),1);
        else
            centers(k,:) = X(randi(m),:);
        end
    end
    J_hist(it) = sum(minD);
    if it>1 && abs(J_hist(it)-J_hist(it-1))<tol
        J_hist = J_hist(1:it);
        break;
    end
end
end
