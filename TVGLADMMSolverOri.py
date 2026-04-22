import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

class TVGLADMMOri:
    def __init__(self, lambda_val: float, beta: float, observations: list[np.ndarray], 
                 penalty_type: str = 'l1', 
                 rho: float = 1.0, fit_epochs: int = 1, tol: float = 1e-6):

        self.T = len(observations)
        self.p = observations[0].shape[1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.lambda_val = lambda_val
        self.beta = beta
        self.data_sequence = observations
        self.penalty_type = penalty_type
        self.rho = rho
        self.epochs = fit_epochs
        self.tol = tol

        self.empirical_covs, self.ns = self.get_covariance_matrices()

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
        covs = []; ns = []
        for t in range(self.T):
            data_t = self.data_sequence[t]
            ns.append(torch.tensor(data_t.shape[0], dtype=torch.float64, device=self.device))
            # case: (N, p) np.ndarray -> wrap to list for uniformity
            if data_t.shape[0] > 1:
                S = np.cov(data_t.T)
            else:
                S = np.outer(data_t, data_t)
            covs.append(torch.tensor(S, dtype=torch.float64, device=self.device))
        return torch.stack(covs), torch.stack(ns) # (T,p,p), (T,1)

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
        eta = self.ns / (m * self.rho) # Step size parameter: (T,)
        eta = eta.view(self.T, 1, 1) # broadcast (T,1,1)
        # Eigen decomposition
        M = eta * self.empirical_covs - A_sym
        eigvals, Q = torch.linalg.eigh(M) # eigvals:(T,p), Q:(T,p,p)
        new_eigs = -0.5 * eigvals + torch.sqrt(0.25*eigvals**2+eta.squeeze(-1))
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
        self.total_residuals = []; self.primal_residuals = []; self.dual_residuals = []
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
            self.primal_residuals.append(primal_residual)
            self.dual_residuals.append(dual_residual)
            self.total_residuals.append(primal_residual + dual_residual)
            
            if verbose == True and primal_residual < self.tol and dual_residual < self.tol:
                print(f"Converged at iteration {iteration} with primal residual {primal_residual}, dual residual {dual_residual}")
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
    def edge_detection(self, thetas_raw: torch.Tensor | np.ndarray, link_threshold: float=1e-3) -> np.ndarray:
        if isinstance(thetas_raw, torch.Tensor):
            thetas_raw = thetas_raw.cpu().numpy()
        if thetas_raw.ndim == 3:
            thetas_shape = thetas_raw.shape
            As = np.empty((thetas_shape[0], thetas_shape[1], thetas_shape[2]))
            for i, theta in enumerate(thetas_raw):
                mask = np.abs(theta) > link_threshold
                mask = mask.astype(int)
                As[i] = mask * theta
            return As # (T,p,p)
        else: raise ValueError(f"\'thetas_raw\' should has 3 dimensions but the current has only {thetas_raw.ndim} dimensions.")

    def f1_score(self, pred_thetas_raw: torch.Tensor, true_thetas: list[np.ndarray], link_threshold: float=1e-3) -> tuple[list[float], float]:
        """This score is the harmonic mean of the precision and recall."""
        pred_thetas = self.edge_detection(pred_thetas_raw, link_threshold)
        if isinstance(true_thetas, list):
            # calculate f1 scores for each prediction
            TPs = []; FPs = []; FNs = []
            precisions = []; recalls = []; f1s = []
            for true, pred in zip(true_thetas, pred_thetas):
                TP = np.sum((true != 0) & (pred != 0))
                FP = np.sum((true == 0) & (pred != 0))
                FN = np.sum((true != 0) & (pred == 0))
                TPs.append(TP)
                FPs.append(FP)
                FNs.append(FN)
                if TP + FP == 0:
                    precision = 0.0
                else:
                    precision = TP / (TP + FP)
                precisions.append(precision)
                if TP + FN == 0:
                    recall = 0.0
                else:
                    recall = TP / (TP + FN)
                recalls.append(recall)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                f1s.append(f1)
            # calculate a total f1 score for all predictions 
            TP_all = np.sum(TPs)
            FP_all = np.sum(FPs)
            FN_all = np.sum(FNs)
            if TP_all + FP_all == 0:
                precision_all = 0.0
            else:
                precision_all = TP_all / (TP_all + FP_all)
            if TP_all + FN_all == 0:
                recall_all = 0.0
            else:
                recall_all = TP_all / (TP_all + FN_all)
            if precision_all + recall_all == 0:
                f1_all = 0.0
            else:
                f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)
            return f1s, f1_all
        else: raise ValueError("Require round truth precision matrices to be a list of numpy arrays.")

    def temporal_deviation_ratio(self, pred_thetas_raw: torch.Tensor, returnlist: bool=True) -> tuple[list[float] | torch.Tensor, float | torch.Tensor]:
        """This score is the ratio of the temporal deviation at the current time to the average
        temporal deviation value across all time stamps."""
        cur_TD_ratios = torch.norm(pred_thetas_raw[1:] - pred_thetas_raw[:-1], "fro", dim=(-2,-1))
        total_TD_ratio = cur_TD_ratios.sum()
        if total_TD_ratio == 0:
            TD_ratio = cur_TD_ratios * 0
        else:
            TD_ratio = cur_TD_ratios / total_TD_ratio
        if returnlist == True:
            return TD_ratio.tolist(), TD_ratio.mean().item()
        else: return TD_ratio, TD_ratio.mean()



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
    data_sequence, true_thetas = create_synthetic_data(T=T, p=10, n_samples=250, change_point=change_point, multi_samples=False)

    start = time.perf_counter()
    print("Training model.")
    print("------------------------------")
    model = TVGLADMMOri(
        lambda_val=10, 
        beta=0,
        observations=data_sequence,
        penalty_type='l2',
        rho=1,
        fit_epochs=250,
        tol=1e-6
    )
    model.fit()
    pred_thetas = model.Thetas
    total_residuals = model.total_residuals
    primal_residuals = model.primal_residuals
    dual_residuals = model.dual_residuals

    TD_ratios, TD_ratios_mean = model.temporal_deviation_ratio(pred_thetas)
    f1s, f1_all = model.f1_score(pred_thetas, true_thetas, link_threshold=1e-1)
    Ts = [str(t)+"-"+str(t+1) for t in range(T-1)]
    print(f"F_1 score for all predictions is {f1_all}.")

    # Plot loss, TD ratios and edge evolution
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(total_residuals, label="total")
    plt.plot(primal_residuals, label="primal")
    plt.plot(dual_residuals, label="dual")
    plt.title('Residuals vs. epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Residuals')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(Ts, TD_ratios, label="msTVGL")
    plt.axhline(y=TD_ratios_mean, linestyle='--', linewidth=1, color='red')
    plt.title('Temporal deviation ratios')
    plt.xlabel('Time shifts')
    plt.ylabel('TD ratios')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot([t+1 for t in range(T)], f1s, label="msTVGL")
    plt.axhline(y=f1_all, linestyle='--', linewidth=1, color='red')
    plt.title('F1 scores')
    plt.xlabel('Time labels')
    plt.ylabel('F1 scores')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # plot heat map of all thetas
    plt.figure(figsize=(15, 5))

    pred_thetas = model.edge_detection(pred_thetas, link_threshold=1e-1)
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