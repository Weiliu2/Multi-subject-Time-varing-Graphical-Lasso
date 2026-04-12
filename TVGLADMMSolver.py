import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

class TVGLADMM:
    def __init__(self, lambda_val: float, beta: float, observations: list[np.ndarray] | list[list[np.ndarray]] | None, 
                 covariance_sum=None, Ns=None, penalty_type: str = 'l1', 
                 rho: float = 1.0, fit_epochs: int = 1, tol: float = 1e-6):

        if observations != None:
            self.T = len(observations)
            if isinstance(observations[0], np.ndarray):
                self.p = observations[0].shape[1]
            elif isinstance(observations[0], list):
                self.p = observations[0][0].shape[1]
            else:
                raise ValueError("\'observations\' must be list[np.ndarray] or list[list[np.ndarray]].")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.lambda_val = lambda_val
        self.beta = beta
        self.data_sequence = observations
        self.penalty_type = penalty_type
        self.rho = rho
        self.epochs = fit_epochs
        self.tol = tol

        if covariance_sum == None and Ns == None and observations != None:
            self.empirical_covs_sum, self.Ns = self.get_covariance_matrices()
        elif covariance_sum != None and Ns != None and observations == None:
            if isinstance(covariance_sum, torch.Tensor) and isinstance(Ns, torch.Tensor):
                self.T = covariance_sum.shape[0]
                self.p = covariance_sum.shape[1]
                self.empirical_covs_sum = covariance_sum
                self.Ns = Ns
                self.empirical_covs_sum = self.empirical_covs_sum.to(dtype=self.dtype, device=self.device)
                self.Ns = self.Ns.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError("\'covariance_sum\' and \'Ns\' must be tensors with shapes (T,p,p) and (T,1,1).")
        else:
            raise ValueError("Input \'covariance_sum\' and \'Ns\' and set \'observations\'==None, or set \'covariance_sum\' and \'Ns\' as None and input \'observations\'.")

        # initialize Thetas and auxiliary parameters
        self.Thetas = torch.zeros((self.T, self.p, self.p), dtype=self.dtype, device=self.device) + torch.eye(self.p, dtype=self.dtype, device=self.device)
        self.Z0 = torch.zeros((self.T, self.p, self.p), dtype=self.dtype, device=self.device)
        self.Z1 = torch.zeros((self.T-1, self.p, self.p), dtype=self.dtype, device=self.device)
        self.Z2 = torch.zeros((self.T-1, self.p, self.p), dtype=self.dtype, device=self.device)
        self.U0 = torch.zeros((self.T, self.p, self.p), dtype=self.dtype, device=self.device)
        self.U1 = torch.zeros((self.T-1, self.p, self.p), dtype=self.dtype, device=self.device)
        self.U2 = torch.zeros((self.T-1, self.p, self.p), dtype=self.dtype, device=self.device)

    def get_covariance_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sum of empirical covariance of all subjects for each time stamp and stack them into a tensor."""
        covs_sum = []; Ns = []
        for t in range(self.T):
            data_t = self.data_sequence[t]
            # case: (N, p) np.ndarray -> wrap to list for uniformity
            if isinstance(data_t, np.ndarray):
                data_t = [data_t]
            covs_t_temp = []
            for data in data_t:
                if data.shape[0] > 1:
                    S = np.cov(data.T)
                else:
                    S = np.outer(data, data)
                covs_t_temp.append(torch.tensor(S, dtype=torch.float64, device=self.device))
            Ns.append(torch.tensor(len(covs_t_temp), dtype=self.dtype, device=self.device))
            covs_t_sum = torch.sum(torch.stack(covs_t_temp), dim=0)
            covs_sum.append(covs_t_sum)         # list of (p,p) tensors
        return torch.stack(covs_sum), torch.stack(Ns).view(self.T,1,1) # (T,p,p), (T,1,1)

    def _od_soft_threshold(self, X: torch.Tensor, threshold: float) -> torch.Tensor:
        """off-diagonal Element-wise soft thresholding."""
        results = torch.zeros_like(X)
        mask = torch.abs(X) > threshold
        results[mask] = torch.sign(X[mask]) * (torch.abs(X[mask]) - threshold)
        results.diagonal(dim1=-2, dim2=-1).copy_(X.diagonal(dim1=-2, dim2=-1))
        return results
    
    def _soft_threshold(self, X: torch.Tensor, threshold: float) -> torch.Tensor:
        """Element-wise soft thresholding."""
        results = torch.zeros_like(X)
        mask = torch.abs(X) > threshold
        results[mask] = torch.sign(X[mask]) * (torch.abs(X[mask]) - threshold)
        return results
    
    def _group_lasso_threshold(self, X: torch.Tensor, threshold: float) -> torch.Tensor:
        """Group lasso thresholding (column-wise)."""
        results = torch.zeros_like(X)
        col_norms = torch.norm(X, dim=1).view(self.T-1, 1, self.p) # (self.T-1, self.p) -> (self.T-1, 1, self.p)
        mask = col_norms > threshold # (self.T-1, 1, self.p)
        mask_expd = mask.expand(-1, self.p, -1) # (self.T-1, self.p, self.p)
        results = (1 - threshold / col_norms) * X * mask_expd
        return results

    def _Laplacian_regularization(self, X: torch.Tensor, threshold: float) -> torch.Tensor:
        """Element-wise Laplacian regularization"""
        results = 1 / (1+2*threshold) * X
        return results

    def _update_theta(self) -> torch.Tensor:
        """Update Theta using proximal operator."""
        # Compute A
        A = torch.zeros((self.T, self.p, self.p), dtype=self.dtype, device=self.device) # (T,p,p)
        A[1:-1] = ((self.Z0[1:-1] - self.U0[1:-1]) + (self.Z1[1:] - self.U1[1:]) + (self.Z2[:-1] - self.U2[:-1])) / 3.0 # Interior nodes
        A[0] = ((self.Z0[0] - self.U0[0]) + (self.Z1[0] - self.U1[0])) / 2.0 # Left boundary
        A[-1] = ((self.Z0[-1] - self.U0[-1]) + (self.Z2[-1] - self.U2[-1])) / 2.0 # Right boundary
        A_sym = (A + A.transpose(-2, -1)) / 2 # Ensure symmetry
        m = torch.ones(self.T, dtype=self.dtype, device=self.device) * 3
        m[0] = 2
        m[-1] = 2
        eta = 1.0 / (m * self.rho) # Step size parameter: (T,)
        eta = eta.view(self.T, 1, 1) # broadcast (T,1,1)
        # Eigen decomposition
        M = eta * self.empirical_covs_sum - A_sym
        eigvals, Q = torch.linalg.eigh(M) # eigvals:(T,p), Q:(T,p,p)
        new_eigs = -0.5 * eigvals + torch.sqrt(0.25*eigvals**2+(self.Ns*eta).squeeze(-1))
        Thetas_new = Q @ torch.diag_embed(new_eigs) @ Q.transpose(-1, -2)
        return (Thetas_new + Thetas_new.transpose(-2,-1)) / 2
    
    def _update_z0(self) -> torch.Tensor:
        """Update Z0 using off-diagonal soft-thresholding."""
        X = self.Thetas + self.U0
        threshold = self.lambda_val / self.rho
        return self._od_soft_threshold(X, threshold)
    
    def _update_z1z2(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update Z1, Z2 for l1, l2 and Laplacian temporal penalty."""
        avg = (self.Thetas[:-1] + self.Thetas[1:] + self.U1 + self.U2) / 2
        diff = self.Thetas[1:] - self.Thetas[:-1] + self.U2 - self.U1
        # Apply proximal operator based on penalty type
        if self.penalty_type == 'l1':
            E = self._soft_threshold(diff, 2 * self.beta / self.rho)
        elif self.penalty_type == 'l2':
            E = self._group_lasso_threshold(diff, 2 * self.beta / self.rho)
        elif self.penalty_type == 'Laplacian':
            E = self._Laplacian_regularization(diff, 2 * self.beta / self.rho)
        else:
            E = self._soft_threshold(diff, 2 * self.beta / self.rho)  # Default to l1
        Z1 = avg - E / 2
        Z2 = avg + E / 2
        return Z1, Z2

    def _update_u0u1u2(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U0 = self.U0 + self.Thetas - self.Z0
        U1 = self.U1 + self.Thetas[:-1] - self.Z1
        U2 = self.U2 + self.Thetas[1:] - self.Z2
        return U0, U1, U2

    def _compute_primal_residual(self) -> float:
        """Compute primal residual for convergence checking."""
        residuals = []
        # Z0 constraints
        residuals.append(torch.norm(self.Thetas - self.Z0, p='fro', dim=(-2,-1)))
        # Z1, Z2 constraints
        residuals.append(torch.norm(self.Thetas[:-1] - self.Z1, p='fro', dim=(-2,-1)))
        residuals.append(torch.norm(self.Thetas[1:] - self.Z2, p='fro', dim=(-2,-1)))
        residuals = torch.cat(residuals)
        return torch.norm(residuals).item()
    
    def _compute_dual_residual(self, Thetas_prev: torch.Tensor) -> float:
        """Compute dual residual for convergence checking."""
        residuals = torch.norm(self.Thetas - Thetas_prev, p='fro')
        return residuals.item()
    
    def fit(self, verbose: bool = True):
        for iteration in range(self.epochs):

            # Store previous Theta for convergence check
            Thetas_prev = self.Thetas

            # Update variables
            self.Thetas = self._update_theta()
            self.Z0 = self._update_z0()
            self.Z1, self.Z2 = self._update_z1z2()
            self.U0, self.U1, self.U2 = self._update_u0u1u2()
    
            # Check convergence
            primal_residual = self._compute_primal_residual()
            dual_residual = self._compute_dual_residual(Thetas_prev)
            self.losses = primal_residual + dual_residual
            
            if verbose == True and primal_residual < self.tol and dual_residual < self.tol:
                print(f"Converged at iteration {iteration}")
                break
                
            if verbose == True and (iteration % 100 == 0 or iteration == self.epochs - 1):
                print(f"Iteration {iteration}, Primal: {primal_residual:.6f}, Dual: {dual_residual:.6f}")

    """Score computation for hyperparameter selection"""     
    def calculate_average_degree(self, thr: float=1e-6):
        """Calculate average interacting node number of each node"""
        final_thetas = self.Thetas
        nzeros_mask = final_thetas.abs() >= thr
        return nzeros_mask.sum() / self.T / self.p 
    
    def Jaccard_similarity(self, thr: float=1e-6):
        """Calculate Jaccard similarity between adjacent graphs"""
        final_thetas= self.Thetas
        nzeros_mask = final_thetas.abs() >= thr
        graph_last = nzeros_mask[:-1]
        graph_curr = nzeros_mask[1:]
        intersec = graph_last * graph_curr
        union = (graph_last + graph_curr) > 0
        return (intersec.sum(dim=(-2,-1)) / union.sum(dim=(-2,-1))).mean()

    """Rough edge construction, edge prediction performance (f1 score), temporal deviation ratio"""
    def tensor2list(self, tensor_dt: torch.Tensor) -> list[np.ndarray] | np.ndarray:
        if tensor_dt.dim() == 3:
            T = tensor_dt.shape[0]
            # cuda device type tensor can't be converted to numpy because cpu is the host memory -> .cpu()
            # can't call numpy() on Tensor that requires grad -> .detach()
            list_dt = tensor_dt.cpu().numpy() 
            return [list_dt[i].squeeze() for i in range(T)]
        elif tensor_dt.dim() == 2:
            return tensor_dt.cpu().numpy()
        else: raise ValueError("Error detected on the shape of input tensor.")

    def edge_detection(self, thetas_raw: torch.Tensor, link_threshold=0.1) -> list[np.ndarray] | np.ndarray:
        thetas = self.tensor2list(thetas_raw)
        if isinstance(thetas, np.ndarray):
            mask = np.abs(thetas) > link_threshold
            mask = mask.astype(int)
            return mask * thetas
        elif isinstance(thetas, list):
            As = []
            for theta in thetas:
                mask = np.abs(theta) > link_threshold
                mask = mask.astype(int)
                As.append(mask * theta)
            return As
        else: raise ValueError("Raw thetas should be numpy array or a list of numpy arrays.")

    def f1_score(self, pred_theta_raw: torch.Tensor, true_theta: list[np.ndarray] | np.ndarray) -> float:
        """This score is the harmonic mean of the precision and recall."""
        pred_theta = self.edge_detection(pred_theta_raw)
        if isinstance(pred_theta, np.ndarray) and isinstance(true_theta, np.ndarray):
            TP = np.sum((true_theta != 0) & (pred_theta != 0))
            FP = np.sum((true_theta == 0) & (pred_theta != 0))
            FN = np.sum((true_theta != 0) & (pred_theta == 0))
            if TP + FP == 0:
                precision = 0.0
            else:
                precision = TP / (TP + FP)
            if TP + FN == 0:
                recall = 0.0
            else:
                recall = TP / (TP + FN)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return f1
        elif isinstance(pred_theta, list) and isinstance(true_theta, list):
            TP, FP, FN = 0, 0, 0
            for true, pred in zip(true_theta, pred_theta):
                TP += np.sum((true != 0) & (pred != 0))
                FP += np.sum((true == 0) & (pred != 0))
                FN += np.sum((true != 0) & (pred == 0))
            if TP + FP == 0:
                precision = 0.0
            else:
                precision = TP / (TP + FP)
            if TP + FN == 0:
                recall = 0.0
            else:
                recall = TP / (TP + FN)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return f1
        else: raise ValueError("Both ground truth Thetas and predictions must be numpy array or a list of numpy arrays.")

    def temporal_deviation_ratio(self, pred_thetas_raw: torch.Tensor, returnlist: bool=True) -> tuple[list[float] | torch.Tensor, float | torch.Tensor]:
        """This score is the ratio of the temporal deviation at the current time to the average
        temporal deviation value across all time stamps."""
        cur_TD_ratios = torch.norm(pred_thetas_raw[1:] - pred_thetas_raw[:-1], "fro", dim=(-2,-1))
        total_TD_ratio = cur_TD_ratios.sum()
        TD_ratio = cur_TD_ratios / total_TD_ratio
        if returnlist == True:
            return TD_ratio.tolist(), TD_ratio.mean().item()
        else: return TD_ratio, TD_ratio.mean()
    

def covariance_stacker(covariance_matrices: list[np.ndarray], Ns: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the sum of all sample covariance matrices and reshape the list of subject numbers per time stamp.
    
    Parameters
    ----------
    covariance_matrices : list[np.ndarray]
        The list of all sample covariance matrices. 
        The length of the outer list is the sum of the numbers of subjects for all time stamps.
        The matrices must be given in the way matching the order of time, 
        for instance the first 3 are from time stamp 1 which matches the first element 3 of Ns...
    Ns : list[int]
        The list of numbers of subjects for all time stamps.

    Return
    ------
    tuple[torch.Tensor, torch.Tensor]

    The first tensor is the sum of covariance matrices with shape (T,p,p) and the second tensor is the reshaped subject numbers with shape (T,1,1).
    """
    if np.sum(Ns) != len(covariance_matrices):
        raise ValueError("Number of covariances doesn't match the total number of samples.")
    cov_sums = []
    loc = 0
    for N in Ns:
        covs = covariance_matrices[loc: loc+N]
        cov_sum_temp = np.sum(covs, axis=0)
        cov_sums.append(cov_sum_temp)
        loc += N
    return torch.tensor(cov_sums), torch.tensor(Ns).view(len(Ns),1,1)

def hyperparameter_tuner(observations: list[np.ndarray] | list[list[np.ndarray]] | None,
                         covariance_sum: torch.Tensor | None=None,
                         Ns_reshape: torch.Tensor | None=None,
                         lambda_range: list[float] | np.ndarray = np.logspace(-4, 0, 10),
                         penalty_type: str = 'l1',
                         ad_min_rate : float = 0.05,
                         ad_max_rate : float = 0.15,
                         thr : float = 1e-3,
                         epoch: int = 100,
                         verbose: bool = True) -> dict | list[dict]:
    """Hyperparameter selection with warm start.
    
    Parameters
    ----------
    observations : list[np.ndarray] | list[list[np.ndarray]]
        The observed samples.
    covariance_sum : torch.Tensor | None
        The stacked sum of covariance matrices of all the time stamps, with shape (T,p,p), defaulted to None
    Ns_reshaped : torch.Tensor | None
        The stacked subject numbers of all the time stamps, with shape (1,p,p), defaulted to None
    lambda_range : list[float] | np.ndarray
        The range of sparsity penalty, defaulted to np.logspace(-4, 0, 10).
    penalty_type : str
        The type of sparsity penalty, chosen between 'l1' and 'l2', defaulted to 'l1'
    ad_min_rate : float
        The minimal percentage of average degree accounting for the total number of nodes, defaulted to 0.05
    ad_max_rate : float
        The maximal percentage of average degree accounting for the total number of nodes, defaulted to 0.15
    thr : float
        The minimal value of a precision matrix entry that can be seen as none zero, defaulted to 1e-3
    epoch : int
        The total iterations for each parameter combination test, defaulted to 100
    verbose : bool
        Whether to display the tunning process or not, defaulted to True
    
    Return
    ------
    A dictionary

    "best lambda" : all sparsity penalties that meet the requirement
    "average degree" " the average degrees for all best parameter combinations
    "best beta" : all temporal penalties that meet the requirement
    "sharp transitions" : the sharp transitions for all best parameter combinations
        We say a transition is sharp if its TD ratio is larger than 1.25 times of the average.
    "Jaccard similarity" : the Jaccard similarity index for all best parameter combinations
    """
    # precompute covariance matrices, ns and mask
    if observations != None and covariance_sum == None and Ns_reshape == None:
        tmp_model = TVGLADMM(lambda_val=0, beta=0, observations=observations)
        empirical_covs_sum = tmp_model.empirical_covs_sum
        Ns = tmp_model.Ns
    elif observations == None and covariance_sum != None and Ns_reshape != None:
        empirical_covs_sum = covariance_sum
        Ns = Ns_reshape
    else:
        raise ValueError("Either input observations and set covariance_sum and Ns_reshape to None or input covariance_sum and Ns_reshape and set observations to None.") 
    results = []
    best_lambda = None
    best_beta = None
    ad_min = ad_min_rate * empirical_covs_sum.shape[-1] 
    ad_max = ad_max_rate * empirical_covs_sum.shape[-1]
    for i, lambda_val in enumerate(lambda_range):
        if verbose == True:
            print("-"*20)
            print(f"Testing lambda={lambda_val:.4f}")
            score = torch.inf
            best_ad = torch.tensor(0)
            best_js = torch.tensor(0)
            # warm start
            for beta_val in lambda_range[:i+1]:
                model = TVGLADMM(lambda_val=lambda_val, 
                                    beta=beta_val, 
                                    observations=observations, 
                                    penalty_type=penalty_type,
                                    fit_epochs=epoch,
                                    covariance_sum=empirical_covs_sum,
                                    Ns=Ns,)
                model.fit(verbose=False)
                # compute number of sharp transitions
                final_thetas = model.Thetas
                TD_ratios, TD_ratios_mean = model.temporal_deviation_ratio(final_thetas, returnlist=False)
                sharp_trans = TD_ratios > TD_ratios_mean*1.25
                count = sharp_trans.sum()
                # compute average degree and Jaccard similarity
                ad = model.calculate_average_degree(thr)
                js = model.Jaccard_similarity(thr)
                if verbose == True:
                    print(f"Sharp transitions for beta={beta_val:.4f} is {count}, average degree is {ad} and Jaccard similarity is {js}")
                # check requirements
                if 0 < count < score and ad_min <= ad <= ad_max and 0.35 <= js <= 0.8:
                    best_lambda = lambda_val
                    best_beta = beta_val
                    score = count
                    best_ad = ad
                    best_js = js
                    results.append({
                    "best lambda": best_lambda,
                    "average degree": best_ad.item(),
                    "best beta": best_beta,
                    "sharp transitions": score.item() if type(score)!=float else score,
                    "Jaccard similarity": best_js.item()
                    })
    return results


"""---------------------------------------Example usage---------------------------------------"""
if __name__ == "__main__":
    
    def create_synthetic_data(T: int, p: int, n_samples: int, change_point: int, multi_samples: bool=True):
        """Generate a series of graphs changing abruptly at an intermediate time stamp."""
        np.random.seed(42)
        Theta1 = np.eye(p)
        Theta2 = np.eye(p)  
        for i in range(0, p-2, 2):
            Theta2[i, i+2] = Theta2[i+2, i] = 0.5
            if i+4 <= p-2:
                Theta2[i, i+4] = Theta2[i+4, i] = 0.5
        data_sequence = []
        true_thetas = []
        if multi_samples == True:
            for t in range(T):
                if t < change_point:
                    true_theta = Theta1
                else:
                    true_theta = Theta2 
                true_thetas.append(true_theta)
                # Generate samples from Gaussian with this precision matrix
                cov_matrix = np.linalg.inv(true_theta)
                s_temp = []
                for i in range(4):
                    samples = np.random.multivariate_normal(
                        mean=np.zeros(p), 
                        cov=cov_matrix, 
                        size=n_samples + i
                    )
                    s_temp.append(samples)
                data_sequence.append(s_temp)
        else:
            for t in range(T):
                if t < change_point:
                    true_theta = Theta1
                else:
                    true_theta = Theta2 
                true_thetas.append(true_theta)
                # Generate samples from Gaussian with this precision matrix
                cov_matrix = np.linalg.inv(true_theta)
                samples = np.random.multivariate_normal(
                    mean=np.zeros(p), 
                    cov=cov_matrix, 
                    size=n_samples
                )
                data_sequence.append(samples)
        return data_sequence, true_thetas

    T, change_point = 30, 15
    data_sequence, true_thetas = create_synthetic_data(T=T, p=10, n_samples=250, change_point=change_point)
    ts = [i**2 for i in range(T)]

    # Tuning lambda and beta
    start = time.perf_counter()
    print("Start hyperparameter tuning.")
    print("-----------------------------")
    results = hyperparameter_tuner(
        observations = data_sequence,
        penalty_type='l1',
        verbose=True,
        epoch=50,
    )
    # lambda_val = results['best_lambda']
    # beta = results['best_beta']
    # print(f"Best lambda: {lambda_val}, best beta: {beta}.")
    print(pd.DataFrame(results))
    end = time.perf_counter()
    print(f"Time cost for hyperparameter tuning: {end - start:.6f} seconds")
    print("------------------------------")

    start = time.perf_counter()
    print("Training model with best hyperparameters.")
    print("------------------------------")
    model = TVGLADMM(
        lambda_val=1e-2, 
        beta=1e-2,
        observations=data_sequence,
        penalty_type='l2',
        rho=1,
        fit_epochs=500,
        tol=1e-6
    )
    model.fit()
    end = time.perf_counter()
    print(f"Time cost for training single model: {end - start:.6f} seconds")
    print("-------------------------------")

    pred_thetas = model.Thetas
    losses = model.losses

    TD_ratios, TD_ratios_mean = model.temporal_deviation_ratio(pred_thetas)
    f1 = model.f1_score(pred_thetas, true_thetas)
    print(f"F_1 score is {f1}.")

    # Plot loss, TD ratios and edge evolution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(TD_ratios)
    plt.axhline(y=TD_ratios_mean, linestyle='--', linewidth=1, color='red')
    plt.title('Temporal deviation ratios')
    plt.xlabel('Epoch')
    plt.ylabel('TD ratios')
    
    plt.tight_layout()
    plt.show()

    # plot heat map of all thetas
    plt.figure(figsize=(15, 5))

    pred_thetas = model.edge_detection(pred_thetas)
    for i in range(T):
        plt.subplot(6, 5, i+1)
        plt.imshow(np.abs(pred_thetas[i]) > 0, cmap='Blues')
        plt.title(f'Sparsity Pattern at t={i}')
        plt.colorbar()

        # plt.subplot(2, 3, 4)
        # plt.imshow(np.abs(pred_thetas[change_point-1]) > 0, cmap='Blues')
        # plt.title(f'Sparsity Pattern at t={change_point-1}')
        # plt.colorbar()

        # plt.subplot(2, 3, 5)
        # plt.imshow(np.abs(pred_thetas[change_point]) > 0, cmap='Blues')
        # plt.title(f'Sparsity Pattern at t={change_point}')
        # plt.colorbar()

        # plt.subplot(2, 3, 6)
        # plt.imshow(np.abs(pred_thetas[change_point+1]) > 0, cmap='Blues')
        # plt.title(f'Sparsity Pattern at t={change_point+1}')
        # plt.colorbar()
    
    plt.tight_layout()
    plt.show()