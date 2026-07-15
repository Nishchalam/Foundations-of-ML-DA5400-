%% Q2: You are given a data-set in the file A2Q2Data train.csv with 10000 points in (R100 , R)
% (Each row corresponds to a datapoint where the first 100 components are features and
% the last component is the associated y value).
% Loads data, computes analytic least-squares, GD, SGD, ridge with CV.
rng(0);

%% load
train = readmatrix('A2Q2Data_train .csv');   % N x 101
test  = readmatrix('A2Q2Data_test.csv');
Xtr = train(:,1:100); ytr = train(:,101);
Xte = test(:,1:100);  yte = test(:,101);

% include intercept
Xtr = [ones(size(Xtr,1),1), Xtr];   % N x (D+1)
Xte = [ones(size(Xte,1),1), Xte];
[N, Dp] = size(Xtr);                % Dp = 101

%% (i)Obtain the least squares solution wM L to the regression problem using the ana-
% lytical solution.
wML = (Xtr' * Xtr) \ (Xtr' * ytr);  % analytic solution

%% (ii) Code the gradient descent algorithm with suitable step size to solve the least
% squares algorithms and plot ∥wt − wM L∥2 as a function of t. What do you
% observe?
% minimize f(w) = 0.5 * ||Xw - y||^2
max_iters = 5000;
% Lipschitz constant L = lambda_max(X'X)
opts.eigs_tol = 1e-4;
R = (Xtr' * Xtr);
L = max(eig(R));                     % safe for moderate D
alpha = 1 / (1.1 * L);               % step-size slightly below 1/L
w = zeros(Dp,1);
hist_gd = zeros(max_iters,1);
for t = 1:max_iters
    grad = Xtr' * (Xtr * w - ytr);   % gradient of 0.5||Xw-y||^2
    w = w - alpha * grad;
    hist_gd(t) = norm(w - wML, 2);
    if t>1 && abs(hist_gd(t)-hist_gd(t-1)) < 1e-12
        hist_gd = hist_gd(1:t); break;
    end
end

%% (iii)  Code the stochastic gradient descent algorithm using batch size of 100 and plot
% ∥wt − wM L ∥2 as a function of t. What are your observations?
batch = 100;
epochs = 50;                          % total passes
iters = epochs * ceil(N / batch);
w_sgd = zeros(Dp,1);
hist_sgd = zeros(iters,1);
iter = 0;
eta0 = 0.0001;                       % initial step-size scale
for e = 1:epochs
    idx = randperm(N);
    for b = 1:batch:N
        iter = iter + 1;
        bs = idx(b:min(b+batch-1,N));
        Xb = Xtr(bs, :); yb = ytr(bs);
        gradb = Xb' * (Xb * w_sgd - yb);   % gradient for mini-batch (not normalized)
        eta = eta0 / sqrt(e);              % simple decay by epoch
        w_sgd = w_sgd - eta * gradb;
        hist_sgd(iter) = norm(w_sgd - wML, 2);
    end
end
hist_sgd = hist_sgd(1:iter);

%% (iv) Code the gradient descent algorithm for ridge regression. Cross-validate for var-
% ious choices of λ and plot the error in the validation set as a function of λ. For
% the best λ chosen, obtain wR . Compare the test error (for the test data in the
% file A2Q2Data test.csv) of wR with wM L . Which is better and why?
% closed-form ridge: w = (X'X + lambda*I) \ (X'y)
kfold = 5;
cv_idx = repmat(1:kfold, 1, ceil(N/kfold));
cv_idx = cv_idx(1:N);
cv_idx = cv_idx(randperm(N));   % random folds

lambdas = logspace(-6, 3, 50);
val_err = zeros(length(lambdas),1);
wR_all = zeros(Dp, length(lambdas));

for li = 1:length(lambdas)
    lam = lambdas(li);
    errs = zeros(kfold,1);
    for k = 1:kfold
        tr = (cv_idx ~= k);
        va = (cv_idx == k);
        Xt = Xtr(tr,:); yt = ytr(tr);
        Xv = Xtr(va,:); yv = ytr(va);
        wRk = (Xt' * Xt + lam * eye(Dp)) \ (Xt' * yt);
        errs(k) = mean((Xv * wRk - yv).^2);
    end
    val_err(li) = mean(errs);
    wR_all(:,li) = (Xtr' * Xtr + lam * eye(Dp)) \ (Xtr' * ytr);
end

[~, idx_best] = min(val_err);
lambda_best = lambdas(idx_best);
wR = wR_all(:,idx_best);

% --- Test errors
mse_wML = mean((Xte * wML - yte).^2);
mse_wR  = mean((Xte * wR  - yte).^2);

% --- Plot 1: Validation MSE vs λ
figure(2); clf;
semilogx(lambdas, val_err, 'LineWidth',1.2);
xline(lambda_best,'--r','Best λ');
xlabel('\lambda'); ylabel('Validation MSE');
title('Cross-validation curve for Ridge Regression');
grid on;

% --- Plot 2: Coefficient magnitudes vs λ
figure(3); clf;
semilogx(lambdas, abs(wR_all(2:end,:)), 'LineWidth',0.8);  % skip bias term
xlabel('\lambda'); ylabel('|w_j|');
title('Shrinkage of Ridge Coefficients with λ');
grid on;

fprintf('λ* = %.3e | Test MSE (wML)=%.5f | Test MSE (wR)=%.5f\n', ...
        lambda_best, mse_wML, mse_wR);
