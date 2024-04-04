from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import warnings
from sklearn.exceptions import ConvergenceWarning

##warnings.filterwarnings("ignore", category=ConvergenceWarning)

##########################################
##########################################
##########################################

# Function for generating data
def generate_data(n_samples, p):

    ## case 1
    X = np.random.randn(n_samples, p)
    #y = 1 * X[:, 0] + 1 * X[:, 1] + 1 * X[:, 2] + 0.01 * X[:, 3] + 0.01 * X[:, 4] + 0.01 * X[:, 5] + np.random.randn(n_samples)
    
    ## Case 2
    #X = np.zeros((n_samples, p))
    #mean = np.zeros(p)
    #cov = np.zeros((p, p))
    #for i in range(p):
    #    for j in range(p):
    #        cov[i][j] = 0.5 ** abs(i - j)
    #for i in range(n_samples):
    #    X[i, :] = np.random.multivariate_normal(mean=mean,cov=cov)
    #y = 1 * X[:, 0] + 1 * X[:, 1] + 1 * X[:, 2] + 0.01 * X[:, 3] + 0.01 * X[:, 4] + 0.01 * X[:, 5] + np.random.randn(n_samples)
    y = 1 * X[:, 0] + 1 * X[:, 1] + 1 * X[:, 2] + 1 * X[:, 3] + 1 * X[:, 4] + 1 * X[:, 5] + 1 * X[:, 6] + 1 * X[:, 7] + 1 * X[:, 8] + 1 * X[:, 9] + 1 * X[:, 10] + 0.01 * X[:, 11] + 0.01 * X[:, 12] + 0.01 * X[:, 13] + 0.01 * X[:, 14] + 0.01 * X[:, 15] + 0.01 * X[:, 16] + 0.01 * X[:, 17] + 0.01 * X[:, 18] + 0.01 * X[:, 19] + 0.01 * X[:, 20] + np.random.randn(n_samples)
    return X, y




# Function for generating streaming data
def stream_matrix(matrix, b):

    sub_matrices = np.array_split(matrix, b, axis=0)

    for sub_matrix in sub_matrices:
        yield sub_matrix



### Following Zhang & Zhang, the debiased lasso for D_1
def my_debiased_lasso(X, y):

    n, p = X.shape
    lambda0 = math.sqrt(2 * math.log(p) / n)
    model = sm.OLS(y, X)
    results = model.fit()
    beta_hat = results.params.reshape(-1, 1)  
    beta_hat = np.zeros(p).reshape(-1, 1) 
    for iter_count in range(1000):
        temp = 0
        if iter_count > 1:
            temp = sigma2_hat
        sigma2_hat = (np.linalg.norm(y.reshape(-1, 1) - X @ beta_hat) ** 2) / n
        lambda_hat = math.sqrt(sigma2_hat) * lambda0
        clf = Lasso(alpha=lambda_hat, max_iter=1000, selection='random', tol=1e-6)
        clf.fit(X, y)
        beta_hat = clf.coef_.reshape(-1, 1)
        if abs(sigma2_hat - temp) < 1e-4:
            break
    
    return beta_hat, sigma2_hat



# Debiased lasso for D_1
def debiased_lasso(X, y, alpha):

    n, p = X.shape
    clf = Lasso(alpha=alpha, max_iter=1000, selection='cyclic', tol=1e-6)
    clf.fit(X, y)
    beta_lasso = clf.coef_.reshape(-1, 1)

    gamma = []
    beta_debiased = []
    for r in range(0, p):
        Xr = np.delete(X, r, axis=1)
        xr = X[:, r:r+1]
        clf.fit(Xr, xr)
        temp = clf.coef_.reshape(-1, 1)
        gamma.append(temp)
        z = xr - Xr @ gamma[r]
        a1 = z.T @ xr
        a2 = z.T @ y
        A1 = z.T @ X
        beta_debiased.append(beta_lasso[r, 0] + (a2 - A1 @ beta_lasso) / a1)

    return beta_debiased



# 4.2
# Lasso CV for D_1; return lambda_1   corresponding to (2) in the paper
def lasso_cv(X, y, lambda_values, n_folds=5):
    
    n, p = X.shape
    kf = KFold(n_splits=n_folds)
    cv_scores = np.zeros(len(lambda_values))

    for i, lambda_n in enumerate(lambda_values):
        scores = []
        clf = Lasso(alpha=lambda_n, max_iter=1000, selection='cyclic', tol=1e-6)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            clf.fit(X_train, y_train)
            beta = clf.coef_.reshape(-1, 1)
            y_pred = X_val @ beta
            score = np.mean((y_val.reshape(-1, 1) - y_pred) ** 2)
            scores.append(score)
        cv_scores[i] = np.mean(scores)

    best_lambda = lambda_values[np.argmin(cv_scores)]
    sigma2_lambda = cv_scores[np.argmin(cv_scores)]
    return best_lambda, sigma2_lambda


        
# Debiased lasso CV for D_1; return lambda_1, sigma^2^(1)
def debiased_lasso_cv(X, y, lambda_values, n_folds=5):
    
    n, p = X.shape
    kf = KFold(n_splits=n_folds)
    cv_scores = np.zeros(len(lambda_values))

    for i, lambda_n in enumerate(lambda_values):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            beta = debiased_lasso(X_train, y_train, lambda_n)
            beta = np.array(beta).reshape(p, 1)
            y_pred = X_val @ beta
            score = np.mean((y_val.reshape(-1, 1) - y_pred) ** 2)
            scores.append(score)
        cv_scores[i] = np.mean(scores)

    best_lambda = lambda_values[np.argmin(cv_scores)]
    sigma2_lambda = cv_scores[np.argmin(cv_scores)]
    return best_lambda, sigma2_lambda


#######################################################
#######################################################
#######################framework#######################
#######################################################
#######################################################


(rep, p, s0) = (200, 400, 6)
bias_ols_mean = np.zeros(p)
std_ols_mean = np.zeros(p)
bias_odl_mean = np.zeros(p)
std_odl_mean = np.zeros(p)
bias_odl_2_mean = np.zeros(p)
std_odl_2_mean = np.zeros(p)
bias_odl_4_mean = np.zeros(p)
std_odl_4_mean = np.zeros(p)
bias_odl_6_mean = np.zeros(p)
std_odl_6_mean = np.zeros(p)
bias_odl_8_mean = np.zeros(p)
std_odl_8_mean = np.zeros(p)
bias_odl_10_mean = np.zeros(p)
std_odl_10_mean = np.zeros(p)
lambda_all = np.zeros(p)
for i in range(rep):

    # Generate streaming data
    #np.random.seed(1234)
    (n_samples, p, b) = (420, 100, 12)
    X, y = generate_data(n_samples, p)
    D = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    stream = stream_matrix(D, b)

    # Perform OLS
    model = sm.OLS(y, X)
    results = model.fit()
    beta_true = np.concatenate([np.array([1, 1, 1, 0.01, 0.01, 0.01]), np.zeros(p - s0)])
    #beta_true = np.concatenate([np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), np.zeros(p - s0)])
    bias_ols = abs(results.params - beta_true)
    std_ols = results.bse



    # value of beta^(1), gamma_r^(1)
    n1 = int(n_samples / b)
    X1 = X[0:n1, ]
    y1 = y[0:n1, ]
    T_lambda = [0.15, 0.20, 0.25, 0.30]
    lambda1, sigma2 = lasso_cv(X1, y1, T_lambda)
    print("lambda1", lambda1)
    clf = Lasso(alpha=lambda1, max_iter=1000, selection='cyclic', tol=1e-6)
    clf.fit(X1, y1)
    beta_lasso = clf.coef_.reshape(-1, 1)
    gamma1, beta1 = [], []
    for r in range(0, p):
        Xr = np.delete(X1, r, axis=1)
        xr = X1[:, r:r+1]
        clf.fit(Xr, xr)
        temp = clf.coef_.reshape(-1, 1)
        gamma1.append(temp)
        z = xr - Xr @ gamma1[r]
        a1 = z.T @ xr
        a2 = z.T @ y1
        A1 = z.T @ X1
        beta1.append(beta_lasso[r, 0] + (a2 - A1 @ beta_lasso) / a1) 

    beta1 = np.array(beta1).reshape(p, 1) ## \beta_{off}^{(1)}
    gammas = list(gamma1)  ## \gamma_r^{(1)}
    print("=======================================")


    # beta^(1) at lambda grids             
    beta1lambda = []
    for lam in T_lambda:
        bbeta1 = []
        for r in range(0, p):
            Xr = np.delete(X1, r, axis=1)
            xr = X1[:, r:r+1]
            clf.fit(Xr, xr)
            temp = clf.coef_.reshape(-1, 1)
            gamma1.append(temp)
            z = xr - Xr @ gamma1[r]
            a1 = z.T @ xr
            a2 = z.T @ y1
            A1 = z.T @ X1
            bbeta1.append(beta_lasso[r, 0] + (a2 - A1 @ beta_lasso) / a1) 

        bbeta1 = np.array(bbeta1).reshape(p, 1)  
        beta1lambda.append(bbeta1)



    # Online framework
    #learning_rate = 0.005 
    (j, N) = (0, 0)
    S = np.zeros((p, p))
    U = np.zeros((p, 1))
    lambda_choose = []
    Rs, Ts, a1s, a2s, A1s, b1s = [], [], [], [], [], []
    beta_lasts, betas, beta_on, tau, std = [], [], [], [], []
    for sub_matrix in stream:

        # 2.1 Online lasso
        j = j + 1 
        N = N + sub_matrix.shape[0]
        X = sub_matrix[:, :p]
        y = sub_matrix[:, p:p+1]
        S = S + X.T @ X
        U = U + X.T @ y
        ## j = 1 (offline)
        if j == 1:
            beta = beta1  ## \beta_{off}^{(1)}
            betas = list(beta1lambda) ## \beta_{off}^{(1)} at lambda grids
            print("sigma2 = ", sigma2)
        ## j > 1 (online)
        else:
            beta_lasts = list(betas)
            betas.clear()
            for lambdab in T_lambda:
                bbeta = np.zeros(p).reshape(-1, 1)
                for iter_count in range(1000):
                    grad = (S @ bbeta - U) / N
                    bbeta = bbeta - learning_rate * grad
                    bbeta = np.sign(bbeta) * np.maximum(np.abs(bbeta) - learning_rate * lambdab, 0)
                    if np.linalg.norm(grad, ord='fro') < 1e-6:
                        break
                betas.append(bbeta)



            # 2.4 Tuning parameter selection
            val = []
            for mybeta in beta_lasts:   
                temp = np.linalg.norm(y - X @ mybeta) ** 2
                val.append(temp)
            index = val.index(min(val))
            beta = betas[index]
            lambda_choose.append(T_lambda[index])
            print("choose lambda = ", T_lambda[index])



            ## estimator of sigma^2^(b)    (9) in the paper
            rr = np.linalg.norm(y - X @ beta) ** 2
            sigma2 = (N - sub_matrix.shape[0]) * sigma2 / N + rr / N
            print("sigma2 = ", sigma2)
            


        # 2.2 Online low-dimensional projection
        tau.clear()
        std.clear()
        for r in range(0, p):
            Xr = np.delete(X, r, axis=1)
            xr = X[:, r:r+1]
            R = np.zeros((p-1, p-1))
            T = np.zeros((p-1, 1))

            ## step 1
            if j == 1:
                R = Xr.T @ Xr
                T = Xr.T @ xr
                Rs.append(R)
                Ts.append(T)
            else:
                Rs[r] = Rs[r] + Xr.T @ Xr
                Ts[r] = Ts[r] + Xr.T @ xr
                ggamma = np.zeros(p-1).reshape(-1, 1)
                for iter_count in range(1000):
                    grad = (Rs[r] @ ggamma - Ts[r]) / N
                    ggamma = ggamma - learning_rate * grad
                    ggamma = np.sign(ggamma) * np.maximum(np.abs(ggamma) - learning_rate * lambdab, 0)
                    if np.linalg.norm(grad, ord='fro') < 1e-6:
                        break
                gammas[r] = ggamma

            ## z_r^(j)
            zr = xr - Xr @ gammas[r]



        # 2.3 Online debiased lasso estimator
            ## a_1^(b), a_2^(b), A_1^(b)
            if j == 1:
                a1s.append(zr.T @ xr)
                a2s.append(zr.T @ y)
                A1s.append(zr.T @ X)
                b1s.append(zr.T @ zr)
            else:
                a1s[r] = a1s[r] + zr.T @ xr
                a2s[r] = a2s[r] + zr.T @ y
                A1s[r] = A1s[r] + zr.T @ X
                b1s[r] = b1s[r] + zr.T @ zr

            ## beta_{on,r}^(b)
            if j == 1:
                beta_on.append(beta[r,0] + (a2s[r] - A1s[r] @ beta) / a1s[r])
            else:
                beta_on[r] = beta[r,0] + (a2s[r] - A1s[r] @ beta) / a1s[r]
                
            ## value of tau_r^(b)
            tau.append(math.sqrt(b1s[r]) / a1s[r])

            ## estimated standard error
            std.append(math.sqrt(sigma2) * tau[r])


            
        ## save the vale when b = 2, 4, 6, 8, 10
        if j == 2:
            beta_odl_2 = np.zeros(p)
            std_odl_2 = np.zeros(p)
            for r in range(0, p):
                beta_odl_2[r] = np.array(beta_on)[r]
                std_odl_2[r] = np.array(std)[r]
            bias_odl_2 = abs(beta_odl_2 - beta_true)
        elif j == 4:
            beta_odl_4 = np.zeros(p)
            std_odl_4 = np.zeros(p)
            for r in range(0, p):
                beta_odl_4[r] = np.array(beta_on)[r]
                std_odl_4[r] = np.array(std)[r]
            bias_odl_4 = abs(beta_odl_4 - beta_true)
        elif j == 6:
            beta_odl_6 = np.zeros(p)
            std_odl_6 = np.zeros(p)
            for r in range(0, p):
                beta_odl_6[r] = np.array(beta_on)[r]
                std_odl_6[r] = np.array(std)[r]
            bias_odl_6 = abs(beta_odl_6 - beta_true)
        elif j == 8:
            beta_odl_8 = np.zeros(p)
            std_odl_8 = np.zeros(p)
            for r in range(0, p):
                beta_odl_8[r] = np.array(beta_on)[r]
                std_odl_8[r] = np.array(std)[r]
            bias_odl_8 = abs(beta_odl_8 - beta_true)
        elif j == 10:
            beta_odl_10 = np.zeros(p)
            std_odl_10 = np.zeros(p)
            for r in range(0, p):
                beta_odl_10[r] = np.array(beta_on)[r]
                std_odl_10[r] = np.array(std)[r]
            bias_odl_10 = abs(beta_odl_10 - beta_true)

        ## print b
        print("j=", j)



    # bias ODL & std ODL over the streaming
    beta_odl = np.zeros(p)
    std_odl = np.zeros(p)
    for r in range(0, p):
        beta_odl[r] = np.array(beta_on)[r]
        std_odl[r] = np.array(std)[r]
    bias_odl = abs(beta_odl - beta_true)



    # summary over the replications    
    bias_ols_mean += bias_ols
    std_ols_mean += std_ols
    bias_odl_mean += bias_odl
    std_odl_mean += std_odl
    bias_odl_2_mean += bias_odl_2
    std_odl_2_mean += std_odl_2
    bias_odl_4_mean += bias_odl_4
    std_odl_4_mean += std_odl_4
    bias_odl_6_mean += bias_odl_6
    std_odl_6_mean += std_odl_6
    bias_odl_8_mean += bias_odl_8
    std_odl_8_mean += std_odl_8
    bias_odl_10_mean += bias_odl_10
    std_odl_10_mean += std_odl_10



#######################################################
#######################################################
###################Output the results##################
#######################################################
#######################################################


print("++++++++++++++The Results+++++++++++++++")

bias_ols_mean = bias_ols_mean / rep
std_ols_mean = std_ols_mean / rep
bias_odl_2_mean = bias_odl_2_mean / rep
std_odl_2_mean = std_odl_2_mean / rep
bias_odl_4_mean = bias_odl_4_mean / rep
std_odl_4_mean = std_odl_4_mean / rep
bias_odl_6_mean = bias_odl_6_mean / rep
std_odl_6_mean = std_odl_6_mean / rep
bias_odl_8_mean = bias_odl_8_mean / rep
std_odl_8_mean = std_odl_8_mean / rep
bias_odl_10_mean = bias_odl_10_mean / rep
std_odl_10_mean = std_odl_10_mean / rep
bias_odl_mean = bias_odl_mean / rep
std_odl_mean = std_odl_mean / rep

# results for bais and std for 1, 0.01, 0 for b=2, 4, 6, 8, 10, 12
s = int(s0/2)
bias_ols_1 = np.mean(bias_ols_mean[:s])
bias_ols_2 = np.mean(bias_ols_mean[s:s0])
bias_ols_3 = np.mean(bias_ols_mean[s0:])
std_ols_1 = np.mean(std_ols_mean[:s])
std_ols_2 = np.mean(std_ols_mean[s:s0])
std_ols_3 = np.mean(std_ols_mean[s0:])
bias_odl_2_1 = np.mean(bias_odl_2_mean[:s])
bias_odl_2_2 = np.mean(bias_odl_2_mean[s:s0])
bias_odl_2_3 = np.mean(bias_odl_2_mean[s0:])
std_odl_2_1 = np.mean(std_odl_2_mean[:s])
std_odl_2_2 = np.mean(std_odl_2_mean[s:s0])
std_odl_2_3 = np.mean(std_odl_2_mean[s0:])
bias_odl_4_1 = np.mean(bias_odl_4_mean[:s])
bias_odl_4_2 = np.mean(bias_odl_4_mean[s:s0])
bias_odl_4_3 = np.mean(bias_odl_4_mean[s0:])
std_odl_4_1 = np.mean(std_odl_4_mean[:s])
std_odl_4_2 = np.mean(std_odl_4_mean[s:s0])
std_odl_4_3 = np.mean(std_odl_4_mean[s0:])
bias_odl_6_1 = np.mean(bias_odl_6_mean[:s])
bias_odl_6_2 = np.mean(bias_odl_6_mean[s:s0])
bias_odl_6_3 = np.mean(bias_odl_6_mean[s0:])
std_odl_6_1 = np.mean(std_odl_6_mean[:s])
std_odl_6_2 = np.mean(std_odl_6_mean[s:s0])
std_odl_6_3 = np.mean(std_odl_6_mean[s0:])
bias_odl_8_1 = np.mean(bias_odl_8_mean[:s])
bias_odl_8_2 = np.mean(bias_odl_8_mean[s:s0])
bias_odl_8_3 = np.mean(bias_odl_8_mean[s0:])
std_odl_8_1 = np.mean(std_odl_8_mean[:s])
std_odl_8_2 = np.mean(std_odl_8_mean[s:s0])
std_odl_8_3 = np.mean(std_odl_8_mean[s0:])
bias_odl_10_1 = np.mean(bias_odl_10_mean[:s])
bias_odl_10_2 = np.mean(bias_odl_10_mean[s:s0])
bias_odl_10_3 = np.mean(bias_odl_10_mean[s0:])
std_odl_10_1 = np.mean(std_odl_10_mean[:s])
std_odl_10_2 = np.mean(std_odl_10_mean[s:s0])
std_odl_10_3 = np.mean(std_odl_10_mean[s0:])
bias_odl_1 = np.mean(bias_odl_mean[:s])
bias_odl_2 = np.mean(bias_odl_mean[s:s0])
bias_odl_3 = np.mean(bias_odl_mean[s0:])
std_odl_1 = np.mean(std_odl_mean[:s])
std_odl_2 = np.mean(std_odl_mean[s:s0])
std_odl_3 = np.mean(std_odl_mean[s0:])


print("bias OLS 1: ", bias_ols_1)
print("bias OLS 0.01: ", bias_ols_2)
print("bias OLS 0: ", bias_ols_3)
print("std OLS 1: ", std_ols_1)
print("std OLS 0.01: ", std_ols_2)
print("std OLS 0: ", std_ols_3)
print("--------------------------------------")
print("b=2, bias ODL 1: ", bias_odl_2_1)
print("b=2, bias ODL 0.01: ", bias_odl_2_2)
print("b=2, bias ODL 0: ", bias_odl_2_3)
print("b=2, std ODL 1: ", std_odl_2_1)
print("b=2, std ODL 0.01: ", std_odl_2_2)
print("b=2, std ODL 0: ", std_odl_2_3)
print("--------------------------------------")
print("b=4, bias ODL 1: ", bias_odl_4_1)
print("b=4, bias ODL 0.01: ", bias_odl_4_2)
print("b=4, bias ODL 0: ", bias_odl_4_3)
print("b=4, std ODL 1: ", std_odl_4_1)
print("b=4, std ODL 0.01: ", std_odl_4_2)
print("b=4, std ODL 0: ", std_odl_4_3)
print("--------------------------------------")
print("b=6, bias ODL 1: ", bias_odl_6_1)
print("b=6, bias ODL 0.01: ", bias_odl_6_2)
print("b=6, bias ODL 0: ", bias_odl_6_3)
print("b=6, std ODL 1: ", std_odl_6_1)
print("b=6, std ODL 0.01: ", std_odl_6_2)
print("b=6, std ODL 0: ", std_odl_6_3)
print("---------------------------------------")
print("b=8, bias ODL 1: ", bias_odl_8_1)
print("b=8, bias ODL 0.01: ", bias_odl_8_2)
print("b=8, bias ODL 0: ", bias_odl_8_3)
print("b=8, std ODL 1: ", std_odl_8_1)
print("b=8, std ODL 0.01: ", std_odl_8_2)
print("b=8, std ODL 0: ", std_odl_8_3)
print("---------------------------------------")
print("b=10, bias ODL 1: ", bias_odl_10_1)
print("b=10, bias ODL 0.01: ", bias_odl_10_2)
print("b=10, bias ODL 0: ", bias_odl_10_3)
print("b=10, std ODL 1: ", std_odl_10_1)
print("b=10, std ODL 0.01: ", std_odl_10_2)
print("b=10, std ODL 0: ", std_odl_10_3)
print("---------------------------------------")
print("bias ODL 1: ", bias_odl_1)
print("bias ODL 0.01: ", bias_odl_2)
print("bias ODL 0: ", bias_odl_3)
print("std ODL 1: ", std_odl_1)
print("std ODL 0.01: ", std_odl_2)
print("std ODL 0: ", std_odl_3)


