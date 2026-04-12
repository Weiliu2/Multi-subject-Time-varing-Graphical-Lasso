import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

"""
    TVGL Neural Network solver v2.4
    1.For each time stamp samples are batched
    2.covariance matrices and sample sizes are now stored in batched tensors with masks to eliminate loops in loss computation
    3.fit function now doesn't output final thetas and losses. Uses model.compute_thetas() and model.losses to get them.
    4.Incorporate functions edge detection, f1 score and TD ratios into main class
    5.Add calculate_information_criterion function in the main class and hyperparameter tuner function 
    6.Add time stamps input so that the algorithm can accept asynchronous sequences
    7.Normalize log-likelihood by removing coefficient self.ns. This alleviates over penalization of time stamps with very few observations. 
    8.Add a auto mode for fit function to train the model until convergence
    9.Add learning rate control in hyperparameter_tuner function
    10.Change hyperparameter_tuner function to warm start 
"""

class TVGLNeural(nn.Module):
    def __init__(self, lambda_val: float, beta: float, observations: list[np.ndarray] | list[list[np.ndarray]], ts: list[int], 
                 covariance=None, ns=None, mask=None, penalty_type='laplacian', learning_rate=0.005, fit_epochs=1):
        super().__init__()
        self.T = len(observations)
        if isinstance(observations[0], np.ndarray):
            self.p = observations[0].shape[1]
        elif isinstance(observations[0], list):
            self.p = observations[0][0].shape[1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ts = torch.tensor(ts, dtype=torch.float32, device=self.device)
        self.lambda_val = lambda_val
        self.beta = beta
        self.data_sequence = observations
        self.penalty_type = penalty_type
        self.lr = learning_rate
        self.epochs = fit_epochs
        self.I = torch.eye(self.p, device=self.device)
        self.sparsity_mask = ~torch.eye(self.p, dtype=torch.bool, device=self.device) # ~:invert 0->1 1->0 true->false false->true

        if covariance == None and ns == None and mask == None:
            self.empirical_covs, self.ns, self.mask = self.get_covariance_matrices()
        else:
            self.empirical_covs = covariance
            self.ns = ns
            self.mask = mask

        # Learnable Cholesky-like parameters
        self.L_params = nn.Parameter(0.1 * torch.randn(self.T, self.p, self.p,
                                               dtype=torch.float32,
                                               device=self.device))

    def get_covariance_matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            covs:  (T, M_max, p, p) 
            ns:    (T, M_max)
            mask:  (T, M_max) boolean for valid entries
        """
        covs = []; ns = []; mask = []
        M_max = max(
            len(self.data_sequence[t]) 
            if isinstance(self.data_sequence[t], list) else 1
            for t in range(self.T)
        )
        for t in range(self.T):
            data_t = self.data_sequence[t]
            # case: (N, p) np.ndarray -> wrap to list for uniformity
            if isinstance(data_t, np.ndarray):
                data_t = [data_t]
            covs_t = []; ns_t = []; mask_t = []

            for data in data_t:
                ns_t.append(data.shape[0])
                mask_t.append(True)
                if data.shape[0] > 1:
                    S = np.cov(data.T)
                else:
                    S = np.outer(data, data)
                covs_t.append(torch.tensor(S, dtype=torch.float32, device=self.device))
            # pad to M_max
            while len(covs_t) < M_max:
                covs_t.append(torch.zeros(self.p, self.p, device=self.device))
                ns_t.append(0.0)
                mask_t.append(False)
            covs.append(torch.stack(covs_t))         # list of (M_max,p,p) tensors
            ns.append(torch.tensor(ns_t, device=self.device)) # list of (M_max) tensors
            mask.append(torch.tensor(mask_t, device=self.device)) # list of (M_max) tensors

        return torch.stack(covs), torch.stack(ns), torch.stack(mask) # self.cov:(T,M_max,p,p), self.ns:(T,M_max), self.mask:(T,M_max)

    def compute_thetas(self) -> tuple[torch.Tensor, torch.Tensor]:
        L = torch.tril(self.L_params)

        # stabilize diag and guarantee PD
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-8 # (T,p) dim=2
        # always clone before in place operation
        L = torch.clone(L) # L=L.clone() # generate a space to store new tensor value while storing old tensor value and old gradient
        L.diagonal(dim1=-2, dim2=-1).copy_(diag) # .copy_() a in place operation changing L directly without recording previous value. If no .clone(), the gradient and value does not match

        Thetas = L @ L.transpose(-1, -2) + 1e-10 * self.I # (T,p,p) dim=3
        # logdet via Cholesky structure
        logdets = 2 * torch.sum(torch.log(diag), dim=-1) # (T) dim=1
        return Thetas, logdets
    
    def log_likelihood_loss(self, Thetas: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        Theta_expanded = Thetas[:, None, :, :] # (T,p,p) -> (T,1,p,p)
        logdet_expanded = logdets[:, None] # (T) -> (T,1)
        trace_terms = torch.einsum( # A and B doesn't have to be aligned on the same second dimension because einsum expands the missing dimension at the beginning
            "tmpq,tmpq->tm", 
            self.empirical_covs, # A (T,M_max,p,p)
            Theta_expanded       # B (T,1,p,p)
        ) # (T,M_max)
        # apply mask
        term = -logdet_expanded + trace_terms
        term = term * self.mask # ignore padded entries: boolean value * a = a or 0
        return term.sum()

    def sparsity_penalty(self, Thetas: torch.Tensor) -> torch.Tensor:
        """L1 norm of off-diagonal elements"""
        mask = self.sparsity_mask
        penalty = torch.sum(torch.abs(Thetas[:, mask]))
        return penalty

    def temporal_penalty(self, Thetas: torch.Tensor) -> torch.Tensor:
        delta_t = (self.ts[1:] - self.ts[:-1]).unsqueeze(-1).unsqueeze(-1)
        diff = (Thetas[1:] - Thetas[:-1]) / delta_t

        if self.penalty_type == 'laplacian':
            penalty = torch.sum(delta_t*(diff ** 2))
        elif self.penalty_type == 'l1':
            penalty = torch.sum(delta_t*(torch.abs(diff)))
        elif self.penalty_type == 'l2':
            penalty = torch.sum(delta_t*(torch.norm(diff, dim=1))) # diff.dim()==3, dim=1: column wise l2 norm, iterate along row idx
        elif self.penalty_type == 'linf':
            penalty = torch.sum(delta_t*(torch.max(torch.abs(diff), dim=2)[0]))
        elif self.penalty_type == 'PN':
            # Approximate the node perturbation penalty
            penalty = torch.sum(delta_t*(torch.sqrt(torch.sum(diff ** 2 + diff.transpose(1, 2) ** 2, dim=2))))
        return penalty

    def forward(self) -> torch.Tensor:
        Thetas, logdets = self.compute_thetas()
        likelihood = self.log_likelihood_loss(Thetas, logdets)
        sparsity = self.lambda_val * self.sparsity_penalty(Thetas)
        temporal = self.beta * self.temporal_penalty(Thetas)
        return likelihood + sparsity + temporal

    def fit(self, verbose: bool=True, mode: str="manual", tol: float=1e-10):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.losses = []

        if mode == "manual":
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                loss = self.forward()
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())

                if verbose==True and (epoch % 100 == 0 or epoch == self.epochs - 1):
                    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        elif mode == "auto":
            epoch = 1
            optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            last_loss = loss.item()
            delta_loss = torch.inf
            while np.abs(delta_loss) / np.abs(last_loss) > 1e-15:
                epoch += 1
                optimizer.zero_grad()
                loss = self.forward()
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())      
                delta_loss = loss.item() - last_loss
                last_loss = loss.item() 
                if verbose==True and (epoch % 100 == 0):
                    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            if verbose==True:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            self.total_step = epoch

    
    """Score computation for hyperparameter selection"""
    def calculate_information_criterion(self, criterion: str) -> torch.Tensor:
        """Calculate information criterion after model is trained.
           Only used for network prediction.
        """
        with torch.no_grad():
            final_thetas, final_logdets = self.compute_thetas()
            final_thetas = final_thetas.detach()
            final_logdets = final_logdets.detach()
            log_likelihood = self.log_likelihood_loss(final_thetas, final_logdets)
        
            n_params = torch.tensor(self.p * self.p * self.T, dtype=torch.float32, device=self.device)
            n_total = torch.sum(self.ns)
        if criterion == 'aic':
            return 2 * log_likelihood + 2 * n_params
        elif criterion == 'bic':
            return 2 * log_likelihood + n_params * torch.log(n_total)
        elif criterion == 'ebic':
            # Extended BIC for high-dimensional data
            return 2 * log_likelihood + n_params * torch.log(n_total) + 2 * torch.log(torch.tensor(self.p, dtype=torch.float32, device=self.device)) * n_params
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
    def calculate_average_degree(self, thr: float=1e-6):
        """Calculate average interacting gene number of each gene"""
        with torch.no_grad():
            final_thetas, _ = self.compute_thetas()
            final_thetas = final_thetas.detach()
        nzeros_mask = final_thetas.abs() >= thr
        return nzeros_mask.sum() / self.T / self.p 
    
    def Jaccard_similarity(self, thr: float=1e-6):
        """Calculate Jaccard similarity between adjacent graphs"""
        with torch.no_grad():
            final_thetas, _ = self.compute_thetas()
            final_thetas = final_thetas.detach()
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
            list_dt = tensor_dt.detach().cpu().numpy() 
            return [list_dt[i].squeeze() for i in range(T)]
        elif tensor_dt.dim() == 2:
            return tensor_dt.detach().cpu().numpy()
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


def hyperparameter_tuner(observations: list[np.ndarray] | list[list[np.ndarray]],
                         ts: list[int],
                         lambda_range: list[float] | np.ndarray = np.logspace(-4, 0, 10),
                         beta_range: list[float] | np.ndarray = np.logspace(-4, 0, 10),
                         penalty_type: str = 'l1',
                         criterion: str = 'ebic',
                         ad_min_rate : float = 0.05,
                         ad_max_rate : float = 0.15,
                         thr : float = 1e-3,
                         epoch: int = 100,
                         lr : float = 0.005,
                         verbose: bool = True,
                         mode: str = "manual") -> dict | list[dict]:
    # precompute covariance matrices, ns and mask
    tmp_model = TVGLNeural(lambda_val=0, beta=0, observations=observations, ts=ts)
    empirical_covs = tmp_model.empirical_covs
    ns = tmp_model.ns
    mask = tmp_model.mask
    if criterion == 'aic' or criterion == 'bic' or criterion == 'ebic':
        best_score = np.inf
        best_lambda = None
        best_beta = None
        results = []    
        for lambda_val in lambda_range:
            for beta_val in beta_range:
                if verbose == True:
                    print(f"Testing lambda={lambda_val:.4f}, beta={beta_val:.4f}")
                    # Fit TVGL on full data
                    model = TVGLNeural(lambda_val=lambda_val, 
                                    beta=beta_val, 
                                    observations=observations, 
                                    ts=ts,
                                    penalty_type=penalty_type,
                                    learning_rate = lr,
                                    fit_epochs=epoch,
                                    covariance=empirical_covs,
                                    ns=ns,
                                    mask=mask)
                    model.fit(verbose=False)
                    # Calculate information criterion
                    score = model.calculate_information_criterion(criterion)
                    results.append({
                        'lambda': lambda_val,
                        'beta': beta_val,
                        'score': score
                    })
                    if score < best_score:
                        best_score = score
                        best_lambda = lambda_val
                        best_beta = beta_val     
        return {'best_lambda': best_lambda,
                'best_beta': best_beta,
                'best_score': best_score,
                'all_results': results
                }
    elif criterion == 'transition':
        results = []
        best_lambda = None
        best_beta = None
        ad_min = ad_min_rate * empirical_covs.shape[-1] 
        ad_max = ad_max_rate * empirical_covs.shape[-1]
        for i, lambda_val in enumerate(lambda_range):
            if verbose == True:
                print("-"*20)
                print(f"Testing lambda={lambda_val:.4f}")
                score = torch.inf
                best_ad = torch.tensor(0)
                best_js = torch.tensor(0)
                # warm start
                for beta_val in lambda_range[:i+1]:
                    model = TVGLNeural(lambda_val=lambda_val, 
                                        beta=beta_val, 
                                        observations=observations, 
                                        ts=ts,
                                        penalty_type=penalty_type,
                                        learning_rate=lr,
                                        fit_epochs=epoch,
                                        covariance=empirical_covs,
                                        ns=ns,
                                        mask=mask)
                    model.fit(verbose=False, mode=mode)
                    # compute number of sharp transitions
                    with torch.no_grad():
                        final_thetas, _ = model.compute_thetas()
                        final_thetas = final_thetas.detach()
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
        ts=ts,
        penalty_type='l2',
        criterion='transition',
        verbose=False,
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
    model = TVGLNeural(
        lambda_val=1e-2, 
        beta=1e-2,
        observations=data_sequence,
        ts=ts,
        penalty_type='l2',
        learning_rate=0.05,
        fit_epochs=500
    )
    model.fit()
    end = time.perf_counter()
    print(f"Time cost for training single model: {end - start:.6f} seconds")
    print("-------------------------------")

    pred_thetas, _ = model.compute_thetas()
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