import numpy as np
import time
from sklearn.model_selection import KFold
from numpy.linalg import norm, lstsq
import scipy as sp
from scipy import sparse
from scipy.sparse import csr_matrix

def estimate_nu_easy(C, d):
    value = 1 / (np.sum(C ** 2) / C.shape[1])
    return value

def estimate_nu_hard(C, d):
    # Get matrix dimensions
    m, n = C.shape
    e = np.ones((m, 1))

    # Append -e as last column
    H = np.hstack((C, -e))

    # Downsample if more than 200 rows
    if m < 201:
        H2 = H
        d2 = d
    else:
        r = np.random.rand(m)
        s2 = np.argsort(r)
        H2 = H[s2[:200], :]
        d2 = d[s2[:200]]

    lamda = 1.0

    # Eigen-decomposition of H2*H2'
    u, vu = np.linalg.eig(H2 @ H2.T)  # u = eigenvalues (length p), vu = eigenvectors (m×p)
    p = len(u)

    d2 = d2.reshape(-1, 1)  # ensure column vector
    yt = (d2.T @ vu).ravel()  # shape (p,)

    lamdaO = lamda + 1
    cnt = 0

    # Iterative update loop
    while abs(lamdaO - lamda) > 1e-4 and cnt < 100:
        cnt += 1
        nu1 = 0
        pr = 0
        ee = 0
        waw = 0
        lamdaO = lamda

        for i in range(p):
            nu1 += lamda / (u[i] + lamda)
            pr += u[i] / (u[i] + lamda) ** 2
            ee += u[i] * yt[i] ** 2 / (u[i] + lamda) ** 3
            waw += lamda ** 2 * yt[i] ** 2 / (u[i] + lamda) ** 2

        lamda = nu1 * ee / (pr * waw)

    value = lamda
    if cnt == 100:
        value = 1.0

    return value

def sign_correctness(AA,dd,w,gamma):
    p = np.sign(np.dot(AA, w) - gamma)
    return np.sum(p == dd) / AA.shape[0] * 100

def average_error(AA,dd,w,gamma):
    import numpy as np

    p = np.dot(AA, w) - gamma
    s = np.sum(np.abs(p - dd))
    r = p.size
    return s / r

def rms_error(AA,dd,w,gamma):
    import numpy as np

    p = np.dot(AA, w) - gamma
    r = p.size
    diff = p - dd
    return np.sqrt(np.sum(diff * np.conj(diff)) / r)

def NSVM_Mangasarian(A, d, k=0, nu=0, verbose=False):
    """
    Python implementation of a Mangasarian-style NSVM.
    """

    if nu == -1:
        nu = estimate_nu_easy(A, d)
    elif nu == 0:
        nu = estimate_nu_hard(A, d)

    results = {
        'train_acc': 0,
        'test_acc': 0,
        'train_avg_err': 0,
        'test_avg_err': 0,
        'train_rms_err': 0,
        'test_rms_err': 0,
        'cpu_time': 0,
        'nu': nu
    }

    start_time = time.time()

    if k == 0:
        w, gamma, iters = solve_nsvm_adam(A, d, nu)
        elapsed = time.time() - start_time
        results.update({
            'train_acc': sign_correctness(A, d, w, gamma),
            'train_avg_err': average_error(A, d, w, gamma),
            'train_rms_err': rms_error(A, d, w, gamma),
            'cpu_time': elapsed
        })
        if verbose:
            print(f"Train Acc: {results['train_acc']:.2f}%, Avg Err: {results['train_avg_err']:.4f}, RMS Err: {results['train_rms_err']:.4f}, Time: {elapsed:.2f}s")
        return w, gamma, results

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accs, test_accs = [], []
    train_avg_errs, test_avg_errs = [], []
    train_rms_errs, test_rms_errs = [], []
    times = []

    for train_idx, test_idx in kf.split(A):
        A_train, A_test = A[train_idx], A[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]

        t0 = time.time()
        w, gamma, iters = solve_nsvm_adam(A_train, d_train, nu)
        elapsed = time.time() - t0

        train_accs.append(sign_correctness(A_train, d_train, w, gamma))
        test_accs.append(sign_correctness(A_test, d_test, w, gamma))

        train_avg_errs.append(average_error(A_train, d_train, w, gamma))
        test_avg_errs.append(average_error(A_test, d_test, w, gamma))

        train_rms_errs.append(rms_error(A_train, d_train, w, gamma))
        test_rms_errs.append(rms_error(A_test, d_test, w, gamma))

        times.append(elapsed)

        if verbose:
            print(f"Fold completed in {elapsed:.2f}s, Train Acc: {train_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%")

    results.update({
        'train_acc': np.mean(train_accs),
        'test_acc': np.mean(test_accs),
        'train_avg_err': np.mean(train_avg_errs),
        'test_avg_err': np.mean(test_avg_errs),
        'train_rms_err': np.mean(train_rms_errs),
        'test_rms_err': np.mean(test_rms_errs),
        'cpu_time': np.mean(times)
    })

    if verbose:
        print(f"\nAverage Train Acc: {results['train_acc']:.2f}%, Test Acc: {results['test_acc']:.2f}%")
        print(f"Average Train Avg Err: {results['train_avg_err']:.4f}, Test Avg Err: {results['test_avg_err']:.4f}")
        print(f"Average Train RMS Err: {results['train_rms_err']:.4f}, Test RMS Err: {results['test_rms_err']:.4f}")
        print(f"Average CPU Time: {results['cpu_time']:.2f}s")

    return w, gamma, results


import numpy as np

def standardize_train(A):
    """Return (A_std, mean, std) where std has minimum eps to avoid div0."""
    eps = 1e-9
    mu = np.mean(A, axis=0, keepdims=True)
    sigma = np.std(A, axis=0, keepdims=True)
    sigma = np.maximum(sigma, eps)
    A_std = (A - mu) / sigma
    return A_std, mu, sigma

def standardize_apply(A, mu, sigma):
    return (A - mu) / sigma

def hinge_grad_batch(A_batch, d_batch, w, gamma, nu):
    """
    Returns gradients (grad_w, grad_gamma) for a batch.
    d_batch in {-1, +1}
    Loss: 0.5 * ||w||^2 + nu * mean( max(0, 1 - d*(A w - gamma)) )
    grad_w = w - (nu / B) * sum_{i in viol} d_i * x_i
    grad_gamma = (nu / B) * sum_{i in viol} d_i
    """
    B = A_batch.shape[0]
    margins = d_batch * (A_batch @ w - gamma)          # shape (B,)
    viol = margins < 1
    if np.any(viol):
        # sum contributions of violating samples
        contrib = ((d_batch[viol])[:, None] * A_batch[viol]).sum(axis=0)  # shape (n,)
        grad_w = w - (nu / B) * contrib
        grad_gamma = (nu / B) * d_batch[viol].sum()
    else:
        grad_w = w.copy()
        grad_gamma = 0.0
    return grad_w, grad_gamma, viol.sum()

def adam_step(m, v, grad, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Classic Adam update. Returns (param_update, new_m, new_v)
    param_update is the delta to ADD to parameters (i.e. -lr * ...)
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    step = lr * m_hat / (np.sqrt(v_hat) + eps)
    return -step, m, v

def solve_nsvm_adam(A, d, nu,
                    lr=1e-3,
                    max_iter=5000,
                    batch_size=64,
                    tol=1e-5,
                    weight_decay=0.0,
                    standardize=True,
                    verbose=False,
                    seed=0):
    """
    Adam-based solver for Mangasarian/hinge-style NSVM (convex-ish).
    Returns w (n,), gamma (scalar), iteration count.
    """
    rng = np.random.default_rng(seed)

    # Convert inputs
    A = np.asarray(A, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).ravel()
    m, n = A.shape

    # Standardize features (fit on entire A). In CV, you should fit on train only.
    if standardize:
        A, mu, sigma = standardize_train(A)
    else:
        mu = np.zeros((1, n)); sigma = np.ones((1, n))

    # include bias gamma separately
    w = np.zeros(n, dtype=np.float64)
    gamma = 0.0

    # Adam state for w and gamma
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mg = 0.0; vg = 0.0

    t = 0
    it = 0
    best_loss = np.inf
    stagnation = 0

    for it in range(1, max_iter + 1):
        # sample minibatch
        idx = rng.choice(m, size=min(batch_size, m), replace=False)
        A_batch = A[idx]
        d_batch = d[idx]

        # compute gradient for batch
        grad_w, grad_gamma, viol_count = hinge_grad_batch(A_batch, d_batch, w, gamma, nu)

        # add L2 (weight decay) gradient for w: grad += weight_decay * w
        if weight_decay:
            grad_w = grad_w + weight_decay * w

        # Adam updates
        t += 1
        step_w, mw, vw = adam_step(mw, vw, grad_w, t, lr=lr)
        step_g, mg, vg = adam_step(mg, vg, np.array([grad_gamma]), t, lr=lr)
        # step_g is array shape (1,) -> scalar
        w += step_w
        gamma += float(step_g[0])

        # compute full objective occasionally for stopping & reporting
        if it % 50 == 0 or it == 1:
            margins = d * (A @ w - gamma)
            hinge = np.maximum(0, 1 - margins).mean()
            obj = 0.5 * np.dot(w, w) + nu * hinge
            if verbose:
                print(f"iter {it:5d}: obj={obj:.6e}, hinge_mean={hinge:.6e}, viol_frac={(margins<1).mean():.4f}")
            # early stopping based on small change in objective
            if obj + 1e-12 < best_loss - 1e-9:
                best_loss = obj
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > 200:
                if verbose:
                    print("Stopping for stagnation")
                break

        # small stopping criterion on parameter step size
        if np.linalg.norm(step_w) < tol and abs(step_g[0]) < tol:
            if verbose:
                print(f"Converged at iter {it}")
            break

    # if standardized, return raw w/gamma in original scale
    if standardize:
        # w_orig = w / sigma.T ; gamma_orig = gamma - mu @ w_orig ?
        # careful: original scoring was A @ w_raw - gamma
        # If A_std = (A - mu)/sigma and we trained w on A_std, then:
        # original score: A @ w_raw - gamma_raw
        # and trained score: A_std @ w - gamma = (A - mu)/sigma @ w - gamma
        # so w_raw = w / sigma, and gamma_raw = gamma + (mu/sigma) @ w
        w_orig = w / sigma.ravel()
        gamma_orig = gamma + float((mu / sigma) @ w)
        return w_orig, float(gamma_orig), it
    else:
        return w, float(gamma), it

