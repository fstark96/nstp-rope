"""
SEP-CMA-ES for TT-MoE Router — NSTP v1
Based on TRINITY/Conductor research from Sakana AI.

Key idea: Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
instead of backpropagation to optimize the TT-based router weights.

CMA-ES is a derivative-free optimization algorithm that:
1. Maintains a covariance matrix for the search distribution
2. Adapts the covariance based on successful mutations
3. Works well for non-convex, noisy objective functions

For the TT-MoE router, this could:
- Escape local minima that gradient descent gets stuck in
- Find better routing policies that maximize task performance
- Handle the discrete routing decisions more naturally

Implementation plan:
1. Flatten TT-router weights into a single parameter vector
2. Define fitness function = validation PPL (lower is better)
3. Run CMA-ES to find better router weights
4. Compare against backprop-trained router
"""
import numpy as np

class CMAES:
    """CMA-ES implementation for TT-MoE router optimization."""
    
    def __init__(self, dim, sigma=0.3, population_size=None):
        self.dim = dim
        self.sigma = sigma
        
        # Default population size = 4 + floor(3*log(dim))
        if population_size is None:
            self.lambda_ = int(4 + 3 * np.log(dim))
        else:
            self.lambda_ = population_size
        
        # Strategy parameters
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        
        # Learning rates
        self.cs = (self.mu + 2) / (dim + self.mu + 5)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.cov = np.eye(dim)
        
        # CMA-ES parameters
        self.eff_popsize = 1 / (self.weights**2).sum()
        self.cc = (4 + self.mu_eff) / (dim + 4 + 2 * self.mu_eff)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        
        # Mean of the search distribution
        self.mean = np.zeros(dim)
        
        # Damping parameter
        self.damps = 1 + np.sqrt(self.mu_eff / (dim + 2))
    
    @property
    def mu_eff(self):
        w_sum = self.weights.sum()
        return (self.weights.sum() ** 2) / ((self.weights ** 2).sum() + 1e-10)
    
    def ask(self):
        """Generate lambda candidate solutions."""
        samples = np.random.multivariate_normal(self.mean, self.cov * self.sigma**2, self.lambda_)
        return samples
    
    def tell(self, solutions, fitnesses):
        """Update the search distribution based on fitness."""
        # Sort by fitness (lower is better)
        indices = np.argsort(fitnesses)
        best_mu = solutions[indices[:self.mu]]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(best_mu * self.weights[:, np.newaxis], axis=0)
        
        # Update covariance matrix
        ps_delta = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)) * \
                   (self.mean - old_mean) / self.sigma
        pc_delta = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc)) * \
                   (self.mean - old_mean) / self.sigma
        
        self.ps = ps_delta
        self.pc = pc_delta
        
        # Covariance update
        rank_one = self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]
        rank_mu = np.zeros_like(self.cov)
        for i in range(self.mu):
            diff = (best_mu[i] - old_mean) / self.sigma
            rank_mu += self.weights[i] * diff[:, np.newaxis] @ diff[np.newaxis, :]
        
        self.cov = (1 - self.c1 - self.cmu) * self.cov + \
                   self.c1 * (rank_one + (1 - self.cc) * self.cov) + \
                   self.cmu * rank_mu
        
        # Update step size
        self.sigma *= np.exp((self.ps.norm() / self.eff_popsize - 1) / self.damps)
        
        # Keep covariance positive definite
        self.cov = np.linalg.cholesky(self.cov)
    
    def best_solution(self):
        return self.mean


def optimize_router_fitness(router_weights_flat, model, val_loader, device, num_eval_batches=10):
    """
    Fitness function: evaluate validation PPL with given router weights.
    Lower PPL = better fitness.
    """
    import torch
    import math
    
    # Load router weights into model
    model.router.load_flat_weights(router_weights_flat)
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_eval_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 50257), y.view(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += x.numel()
    
    return total_loss / total_tokens  # Return CE loss, lower is better


def run_sep_cmaes(model, val_loader, device, max_iterations=50):
    """
    Run SEP-CMA-ES to optimize TT-MoE router.
    
    SEP-CMA-ES (Sep-CMA-ES) uses separate rank-1 updates for each dimension
    group, making it more robust for high-dimensional problems.
    
    This is particularly suited for TT-based routers because:
    1. TT weights have structured low-rank structure
    2. CMA-ES can exploit the tensor structure
    3. Evolution-based optimization finds better local minima
    
    Returns:
        Optimized router weights
        Best fitness history
    """
    print("Running SEP-CMA-ES for TT-MoE router optimization...")
    
    # Get flat router weights
    flat_weights = model.router.get_flat_weights()
    dim = len(flat_weights)
    print(f"Router parameter dimension: {dim:,}")
    
    # Initialize CMA-ES
    cmaes = CMAES(dim=dim, sigma=0.1)
    cmaes.mean = flat_weights.copy()
    
    fitness_history = []
    
    for gen in range(max_iterations):
        # Generate candidate solutions
        candidates = cmaes.ask()
        fitnesses = []
        
        for i, candidate in enumerate(candidates):
            fitness = optimize_router_fitness(candidate, model, val_loader, device)
            fitnesses.append(fitness)
        
        # Update distribution
        cmaes.tell(candidates, fitnesses)
        
        # Record best
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        fitness_history.append(best_fitness)
        
        if (gen + 1) % 5 == 0 or gen == 0:
            current_mean_fitness = np.mean(fitnesses)
            print(f"  Gen {gen+1:3d}: Best={best_fitness:.4f}, Mean={current_mean_fitness:.4f}, σ={cmaes.sigma:.4f}")
        
        # Early stopping if converged
        if len(fitness_history) > 10:
            recent = fitness_history[-10:]
            if max(recent) - min(recent) < 1e-4:
                print(f"  Converged at generation {gen+1}")
                break
    
    # Load best weights
    best_weights = cmaes.best_solution()
    model.router.load_flat_weights(best_weights)
    
    print(f"SEP-CMA-ES complete. Final fitness: {fitness_history[-1]:.4f}")
    print(f"Improvement: {fitness_history[0]:.4f} → {fitness_history[-1]:.4f} ({(fitness_history[0]-fitness_history[-1])/fitness_history[0]*100:.1f}%)")
    
    return best_weights, fitness_history


# TODO: Implement in NSTP router
"""
To implement SEP-CMA-ES in NSTP:

1. Add get_flat_weights() and load_flat_weights() to TT-MoE router
2. Create val_loader from WikiText-2
3. Run SEP-CMA-ES for 50 generations
4. Compare with backprop-trained router

This is computationally expensive (each generation = lambda model evaluations)
but could find better routing policies than gradient descent.
"""