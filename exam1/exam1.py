import math
import pandas as pd

# ==========================================
# HELPER MATRIX & VECTOR UTILITIES (From Scratch)
# ==========================================

def mean_vector(X):
    """Calculates the sample mean vector for a dataset X (list of lists)."""
    N = len(X)
    d = len(X[0])
    m = [0.0] * d
    for row in X:
        for j in range(d):
            m[j] += row[j]
    return [m[j] / N for j in range(d)]

def cov_matrix(X, m):
    """Calculates the sample covariance matrix S for dataset X and mean m."""
    N = len(X)
    d = len(m)
    S = [[0.0] * d for _ in range(d)]
    for row in X:
        for i in range(d):
            for j in range(d):
                S[i][j] += (row[i] - m[i]) * (row[j] - m[j])
    for i in range(d):
        for j in range(d):
            S[i][j] /= N
    return S

def mat_det(M):
    """Calculates the determinant of a square matrix M."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = 0.0
    for c in range(n):
        sub_M = [row[:c] + row[c+1:] for row in M[1:]]
        det += ((-1) ** c) * M[0][c] * mat_det(sub_sub_M := sub_M)
    return det

def mat_inverse(M):
    """Calculates the inverse of a square matrix M using Gauss-Jordan elimination."""
    n = len(M)
    # Augment M with Identity matrix
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    
    for i in range(n):
        # Find pivot
        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or near-singular.")
        for j in range(2 * n):
            A[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(2 * n):
                    A[k][j] -= factor * A[i][j]
                    
    return [row[n:] for row in A]

def mat_vec_mult(M, v):
    """Multiplies matrix M (n x d) by vector v (d)."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

def dot_product(v1, v2):
    """Calculates the dot product of two vectors v1 and v2."""
    return sum(x * y for x, y in zip(v1, v2))

def solve_linear_system(A, b):
    """Solves A * w = b using Gaussian elimination."""
    n = len(A)
    # Form augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_row] = M[max_row], M[i]
        
        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot
            
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
                    
    return [M[i][n] for i in range(n)]


# ==========================================
# PROBLEM 1: Full Gaussian Maximum Likelihood Classifier
# ==========================================

def solve_problem_1():
    # Read CSVs using pandas and convert to lists for pure-python math
    df_trn = pd.read_csv('1stExam/cls1_trn.csv', header=None)
    X_trn = df_trn.iloc[:, :-1].values.tolist()
    y_trn = df_trn.iloc[:, -1].astype(int).values.tolist()
    
    df_tst = pd.read_csv('1stExam/cls1_tst.csv', header=None)
    X_tst = df_tst.values.tolist()
    
    classes = sorted(list(set(y_trn)))
    N = len(y_trn)
    d = len(X_trn[0])
    
    priors, means, covs, inv_covs, dets = {}, {}, {}, {}, {}
    
    # 1. Fit ML Parameters
    for c in classes:
        X_c = [X_trn[i] for i in range(N) if y_trn[i] == c]
        priors[c] = len(X_c) / N
        means[c] = mean_vector(X_c)
        covs[c] = cov_matrix(X_c, means[c])
        dets[c] = mat_det(covs[c])
        inv_covs[c] = mat_inverse(covs[c])

    # 2. Discriminant Function
    def discriminant_g(x, c):
        diff = [x[j] - means[c][j] for j in range(d)]
        inv_diff = mat_vec_mult(inv_covs[c], diff)
        mahalanobis_sq = dot_product(diff, inv_diff)
        return -0.5 * mahalanobis_sq - 0.5 * math.log(dets[c]) + math.log(priors[c])

    # 3. Test Predictions & Pandas Output
    test_preds = []
    for x in X_tst:
        scores = {c: discriminant_g(x, c) for c in classes}
        test_preds.append(max(scores, key=scores.get))
        
    df_result = pd.DataFrame(X_tst)
    df_result['Prediction'] = test_preds
    df_result.to_csv('result1.csv', index=False, header=False)


# ==========================================
# PROBLEM 2: Equal Diagonal Covariance ML Classifier
# ==========================================

def solve_problem_2():
    df_trn = pd.read_csv('cls2_trn.csv', header=None)
    X_trn = df_trn.iloc[:, :-1].values.tolist()
    y_trn = df_trn.iloc[:, -1].astype(int).values.tolist()
    
    df_tst = pd.read_csv('cls2_tst.csv', header=None)
    X_tst = df_tst.values.tolist()
    
    classes = sorted(list(set(y_trn)))
    N = len(y_trn)
    d = len(X_trn[0])
    
    priors, means = {}, {}
    
    for c in classes:
        X_c = [X_trn[i] for i in range(N) if y_trn[i] == c]
        priors[c] = len(X_c) / N
        means[c] = mean_vector(X_c)
        
    S_diag = [0.0] * d
    for i in range(N):
        c = y_trn[i]
        for j in range(d):
            S_diag[j] += (X_trn[i][j] - means[c][j]) ** 2
    S_diag = [S_diag[j] / N for j in range(d)]

    def discriminant_g(x, c):
        val = 0.0
        for j in range(d):
            val -= ((x[j] - means[c][j]) ** 2) / (2.0 * S_diag[j])
        return val + math.log(priors[c])

    test_preds = []
    for x in X_tst:
        scores = {c: discriminant_g(x, c) for c in classes}
        test_preds.append(max(scores, key=scores.get))
        
    df_result = pd.DataFrame(X_tst)
    df_result['Prediction'] = test_preds
    df_result.to_csv('result2.csv', index=False, header=False)


# ==========================================
# PROBLEM 3: Univariate 6th-Order Linear Regression
# ==========================================

def solve_problem_3():
    df_trn = pd.read_csv('regrs1_trn.csv', header=None)
    x_train = df_trn.iloc[:, 0].values.tolist()
    y_train = df_trn.iloc[:, 1].values.tolist() if df_trn.shape[1] > 1 else df_trn.iloc[:, 0].values.tolist()
    
    df_tst = pd.read_csv('regrs1_tst.csv', header=None)
    x_tst = df_tst.iloc[:, 0].values.tolist()
    
    degree = 6
    A = [[x ** p for p in range(degree + 1)] for x in x_train]
    
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(degree + 1)] for i in range(degree + 1)]
    ATy = [sum(A[k][i] * y_train[k] for k in range(len(A))) for i in range(degree + 1)]
    w = solve_linear_system(ATA, ATy)
    
    def predict_g(x):
        return sum(w[p] * (x ** p) for p in range(degree + 1))
    
    test_preds = [predict_g(x) for x in x_tst]
    
    df_result = pd.DataFrame(x_tst)
    df_result['Prediction'] = test_preds
    df_result.to_csv('result3.csv', index=False, header=False)


# ==========================================
# PROBLEM 4: Multivariate 3rd-Order Linear Regression
# ==========================================

def solve_problem_4():
    df_trn = pd.read_csv('regrs2_trn.csv', header=None)
    X_trn = df_trn.iloc[:, :-1].values.tolist()
    y_train = df_trn.iloc[:, -1].values.tolist()
    
    df_tst = pd.read_csv('regrs2_tst.csv', header=None)
    X_tst = df_tst.values.tolist()
    
    d = len(X_trn[0])
    
    def get_poly_features(x):
        feats = [1.0]  
        for i in range(d):
            feats.append(x[i])
        for i in range(d):
            for j in range(i, d):
                feats.append(x[i] * x[j])
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(x[i] * x[j] * x[k])
        return feats

    A = [get_poly_features(x) for x in X_trn]
    num_terms = len(A[0])
    
    ATA = [[sum(A[m][i] * A[m][j] for m in range(len(A))) for j in range(num_terms)] for i in range(num_terms)]
    ATy = [sum(A[m][i] * y_train[m] for m in range(len(A))) for i in range(num_terms)]
    
    w = solve_linear_system(ATA, ATy)
    
    def predict_g(x):
        return dot_product(w, get_poly_features(x))
    
    # Calculate R-squared for reporting
    y_mean = sum(y_train) / len(y_train)
    ss_tot = sum((y - y_mean) ** 2 for y in y_train)
    ss_res = sum((y_train[i] - predict_g(X_trn[i])) ** 2 for i in range(len(y_train)))
    r_squared = 1.0 - (ss_res / ss_tot)
    print(f"Problem 4 R-squared: {r_squared:.6f}")

    # Output file per instructions: each row contains [testing data x], [g(x)][cite: 1]
    test_preds = [predict_g(x) for x in X_tst]
    df_result = pd.DataFrame(X_tst)
    df_result['Prediction'] = test_preds
    df_result.to_csv('result4.csv', index=False, header=False)

if __name__ == '__main__':
    solve_problem_1()
    # solve_problem_2()
    # solve_problem_3()
    # solve_problem_4()