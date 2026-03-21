"""
Latent-Space CBF via VLM Embeddings
====================================

This module implements the MORE NOVEL direction: constructing CBFs
directly in VLM feature space, bypassing explicit geometric extraction.

Key idea: Instead of VLM → text → parse → geometry → CBF,
we use VLM → embedding → learned safety boundary in embedding space.

This is a research prototype exploring:
  1. Using CLIP/SigLIP embeddings as the CBF state space
  2. Learning a safety classifier in embedding space
  3. Converting the classifier into a smooth CBF via gradient penalties
  4. Running the CBF-QP filter in the latent space

This connects to:
  - AnySafe (Agrawal et al., ICRA 2026): image-conditioned constraint specification
  - Nakamura et al. (2025): gradient penalties for smooth latent CBFs
  - Brunke et al. (RA-L 2025): semantic safety constraints

Requirements: pip install torch torchvision transformers pillow
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Install: pip install torch torchvision")


# ============================================================================
# SAFETY MARGIN NETWORK — Learned CBF in embedding space
# ============================================================================

if HAS_TORCH:
    class SafetyMarginNetwork(nn.Module):
        """
        Neural network that maps VLM embeddings to a safety margin value.

        Architecture:
            [scene_embedding; held_object_embedding; ee_state] → MLP → h(z)

        h(z) > 0: safe
        h(z) = 0: safety boundary
        h(z) < 0: unsafe

        Key design choices from Nakamura et al. (2025):
        - Gradient penalty during training for Lipschitz smoothness
        - Trained on both safe and unsafe rollouts for coverage
        """

        def __init__(self, scene_embed_dim: int = 512, state_dim: int = 2,
                     hidden_dim: int = 256):
            super().__init__()
            input_dim = scene_embed_dim * 2 + state_dim  # scene + object + state

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Initialize final layer near zero for stable training
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, 1.0)  # Start optimistic (safe)

        def forward(self, scene_embed: torch.Tensor, object_embed: torch.Tensor,
                    state: torch.Tensor) -> torch.Tensor:
            """
            Args:
                scene_embed: (B, D) scene/image embedding
                object_embed: (B, D) held object embedding
                state: (B, state_dim) end-effector state [x, y]

            Returns:
                h: (B, 1) safety margin values
            """
            z = torch.cat([scene_embed, object_embed, state], dim=-1)
            return self.net(z)

        def safety_value_and_gradient(self, scene_embed: torch.Tensor,
                                       object_embed: torch.Tensor,
                                       state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute h(z) and ∂h/∂state for the CBF constraint."""
            state.requires_grad_(True)
            h = self.forward(scene_embed, object_embed, state)
            grad_h = torch.autograd.grad(h.sum(), state, create_graph=True)[0]
            return h, grad_h


    class LatentCBFTrainer:
        """
        Trains the SafetyMarginNetwork from demonstration data.

        Training data format:
        - Safe demonstrations: (scene_image, held_object, ee_trajectory, label=safe)
        - Unsafe demonstrations: (scene_image, held_object, ee_trajectory, label=unsafe)

        Loss = BCE(h(z), safe_label) + λ_gp * gradient_penalty + λ_cbf * cbf_violation
        """

        def __init__(self, model: SafetyMarginNetwork, lr: float = 1e-3,
                     grad_penalty_weight: float = 0.1, cbf_weight: float = 0.5):
            self.model = model
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.grad_penalty_weight = grad_penalty_weight
            self.cbf_weight = cbf_weight

        def compute_loss(self, scene_embeds: torch.Tensor, object_embeds: torch.Tensor,
                         states: torch.Tensor, labels: torch.Tensor,
                         next_states: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            Compute training loss.

            Args:
                scene_embeds: (B, D)
                object_embeds: (B, D)
                states: (B, state_dim)
                labels: (B,) float, 1.0=safe, 0.0=unsafe
                next_states: (B, state_dim) optional, for CBF decrease condition
            """
            states.requires_grad_(True)
            h = self.model(scene_embeds, object_embeds, states).squeeze(-1)

            # --- Classification loss ---
            # Map h to probability via sigmoid, train with BCE
            # Safe states should have h > 0, unsafe states h < 0
            cls_loss = F.binary_cross_entropy_with_logits(h, labels)

            # --- Gradient penalty for Lipschitz smoothness ---
            # Key insight from Nakamura et al.: without this, the safety
            # boundary is too sharp for optimization-based filtering
            grad_h = torch.autograd.grad(h.sum(), states, create_graph=True)[0]
            grad_norm = grad_h.norm(dim=-1)
            # Penalize gradients that are too large (enforce Lipschitz)
            gp_loss = ((grad_norm - 1.0).clamp(min=0) ** 2).mean()

            # --- CBF decrease condition (if we have transitions) ---
            cbf_loss = torch.tensor(0.0)
            if next_states is not None:
                h_next = self.model(scene_embeds, object_embeds, next_states).squeeze(-1)
                # For safe states, h should not decrease too fast: h_next >= h - α*h
                alpha = 0.5
                safe_mask = labels > 0.5
                if safe_mask.any():
                    decrease = h[safe_mask] - alpha * h[safe_mask].clamp(min=0) - h_next[safe_mask]
                    cbf_loss = F.relu(decrease).mean()

            total_loss = cls_loss + self.grad_penalty_weight * gp_loss + self.cbf_weight * cbf_loss

            return {
                "total": total_loss,
                "classification": cls_loss,
                "gradient_penalty": gp_loss,
                "cbf_condition": cbf_loss,
            }

        def train_step(self, batch: dict) -> Dict[str, float]:
            self.model.train()
            self.optimizer.zero_grad()

            losses = self.compute_loss(
                batch["scene_embeds"],
                batch["object_embeds"],
                batch["states"],
                batch["labels"],
                batch.get("next_states"),
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            return {k: v.item() for k, v in losses.items()}


# ============================================================================
# SYNTHETIC DATA GENERATOR for training the latent CBF
# ============================================================================

def generate_synthetic_training_data(n_samples: int = 2000, embed_dim: int = 512,
                                     seed: int = 42) -> dict:
    """
    Generate synthetic training data for the latent CBF.

    In a real system, this would come from:
    1. Teleoperated demonstrations (safe + unsafe)
    2. VLM embeddings of scene images
    3. Robot state logs

    Here we simulate it with:
    - Random scene embeddings (representing different table configurations)
    - Object embeddings for "cup of water" vs "dry sponge"
    - 2D end-effector positions
    - Labels based on ground-truth constraint regions
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    rng = np.random.RandomState(seed)

    # Fixed "scene embedding" (one scene for now)
    scene_embed = rng.randn(embed_dim).astype(np.float32)
    scene_embed = scene_embed / np.linalg.norm(scene_embed)

    # Object embeddings — different for cup vs sponge
    cup_embed = rng.randn(embed_dim).astype(np.float32)
    cup_embed = cup_embed / np.linalg.norm(cup_embed)

    sponge_embed = rng.randn(embed_dim).astype(np.float32)
    sponge_embed = sponge_embed / np.linalg.norm(sponge_embed)

    # Define ground-truth unsafe regions (for cup of water)
    # Laptop at (0.3, 0.0), phone at (0.0, -0.2)
    unsafe_centers = [np.array([0.3, 0.0]), np.array([0.0, -0.2])]
    unsafe_radii = [0.2, 0.12]

    # Generate EE positions uniformly in workspace
    states = rng.uniform([-0.6, -0.4], [0.6, 0.4], size=(n_samples, 2)).astype(np.float32)

    # Label: unsafe if within any constraint region (for cup of water)
    labels_cup = np.ones(n_samples, dtype=np.float32)
    for center, radius in zip(unsafe_centers, unsafe_radii):
        dists = np.linalg.norm(states - center, axis=1)
        labels_cup[dists < radius] = 0.0

    # Label: always safe for sponge
    labels_sponge = np.ones(n_samples, dtype=np.float32)

    # Generate next_states (small random perturbation)
    next_states = states + rng.randn(n_samples, 2).astype(np.float32) * 0.02

    # Create batches
    scene_embeds = np.tile(scene_embed, (n_samples, 1))

    cup_data = {
        "scene_embeds": torch.from_numpy(scene_embeds),
        "object_embeds": torch.from_numpy(np.tile(cup_embed, (n_samples, 1))),
        "states": torch.from_numpy(states),
        "labels": torch.from_numpy(labels_cup),
        "next_states": torch.from_numpy(next_states),
    }

    sponge_data = {
        "scene_embeds": torch.from_numpy(scene_embeds),
        "object_embeds": torch.from_numpy(np.tile(sponge_embed, (n_samples, 1))),
        "states": torch.from_numpy(states),
        "labels": torch.from_numpy(labels_sponge),
        "next_states": torch.from_numpy(next_states),
    }

    return {
        "cup_data": cup_data,
        "sponge_data": sponge_data,
        "scene_embed": scene_embed,
        "cup_embed": cup_embed,
        "sponge_embed": sponge_embed,
    }


# ============================================================================
# LATENT CBF SAFETY FILTER
# ============================================================================

if HAS_TORCH:
    class LatentCBFSafetyFilter:
        """
        Safety filter operating in VLM embedding space.

        At runtime:
        1. Encode scene image → scene_embed (cached, updated periodically)
        2. Encode held object description → object_embed
        3. At each timestep: evaluate h(scene_embed, object_embed, state)
        4. Solve CBF-QP: min ||u - u_cmd||^2 s.t. ∇h·u >= -α(h)
        """

        def __init__(self, model: SafetyMarginNetwork, scene_embed: np.ndarray,
                     object_embed: np.ndarray, alpha_scale: float = 1.0,
                     u_max: float = 0.5):
            self.model = model
            self.model.eval()
            self.scene_embed = torch.from_numpy(scene_embed).unsqueeze(0)
            self.object_embed = torch.from_numpy(object_embed).unsqueeze(0)
            self.alpha_scale = alpha_scale
            self.u_max = u_max

        @torch.no_grad()
        def evaluate_safety(self, state: np.ndarray) -> float:
            """Get current safety margin value."""
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            h = self.model(self.scene_embed, self.object_embed, state_t)
            return h.item()

        def certify(self, state: np.ndarray, u_cmd: np.ndarray) -> Tuple[np.ndarray, dict]:
            """
            Certify control input using the latent CBF.
            """
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)

            # Get h and gradient
            h_val, grad_h = self.model.safety_value_and_gradient(
                self.scene_embed, self.object_embed, state_t
            )
            h = h_val.item()
            gh = grad_h.squeeze(0).detach().numpy()

            # Class-K∞ function
            alpha = self.alpha_scale * max(h, 0.01) ** 2

            # CBF constraint: ∇h · u >= -α(h)
            # If already satisfied, pass through
            if gh @ u_cmd >= -alpha:
                return np.clip(u_cmd, -self.u_max, self.u_max), {
                    "h_value": h, "modified": False, "grad_h": gh
                }

            # Project u_cmd onto feasible set
            # min ||u - u_cmd||^2 s.t. gh^T u >= -alpha
            # Solution: u = u_cmd + max(0, (-alpha - gh^T u_cmd) / ||gh||^2) * gh
            gh_norm_sq = np.dot(gh, gh) + 1e-8
            violation = -alpha - np.dot(gh, u_cmd)
            if violation > 0:
                u_cert = u_cmd + (violation / gh_norm_sq) * gh
            else:
                u_cert = u_cmd.copy()

            u_cert = np.clip(u_cert, -self.u_max, self.u_max)

            return u_cert, {"h_value": h, "modified": True, "grad_h": gh}


# ============================================================================
# DEMO: Train and evaluate latent CBF
# ============================================================================

def run_latent_cbf_demo():
    """
    Full demo of the latent-space CBF approach.
    Trains a safety margin network and runs it as a safety filter.
    """
    if not HAS_TORCH:
        print("ERROR: PyTorch required. Install: pip install torch torchvision")
        return

    print("=" * 70)
    print("Latent-Space CBF Demo (VLM Embedding Approach)")
    print("=" * 70)

    # --- Generate training data ---
    print("\n[1/4] Generating synthetic training data...")
    data = generate_synthetic_training_data(n_samples=3000, embed_dim=128)

    n_safe = data["cup_data"]["labels"].sum().item()
    n_unsafe = len(data["cup_data"]["labels"]) - n_safe
    print(f"  Cup of water: {int(n_safe)} safe, {int(n_unsafe)} unsafe samples")

    # --- Train the safety margin network ---
    print("\n[2/4] Training SafetyMarginNetwork...")
    model = SafetyMarginNetwork(scene_embed_dim=128, state_dim=2, hidden_dim=128)
    trainer = LatentCBFTrainer(model, lr=5e-4, grad_penalty_weight=0.1, cbf_weight=0.3)

    n_epochs = 80
    batch_size = 256
    n_samples = len(data["cup_data"]["states"])

    for epoch in range(n_epochs):
        # Mini-batch training
        indices = torch.randperm(n_samples)[:batch_size]
        batch = {k: v[indices] for k, v in data["cup_data"].items()}
        losses = trainer.train_step(batch)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"total={losses['total']:.4f}, cls={losses['classification']:.4f}, "
                  f"gp={losses['gradient_penalty']:.4f}, cbf={losses['cbf_condition']:.4f}")

    # --- Evaluate ---
    print("\n[3/4] Evaluating learned CBF...")
    model.eval()

    # Create safety filter for cup of water
    filter_cup = LatentCBFSafetyFilter(
        model, data["scene_embed"], data["cup_embed"], alpha_scale=1.0
    )

    # Create safety filter for sponge
    filter_sponge = LatentCBFSafetyFilter(
        model, data["scene_embed"], data["sponge_embed"], alpha_scale=1.0
    )

    # Test at specific points
    test_points = [
        (np.array([0.3, 0.0]), "over laptop"),
        (np.array([0.0, -0.2]), "over phone"),
        (np.array([-0.3, 0.3]), "empty area"),
        (np.array([0.5, 0.0]), "near laptop edge"),
    ]

    print("\n  Safety values (cup of water vs dry sponge):")
    print(f"  {'Location':<20} {'Cup h(x)':<12} {'Sponge h(x)':<12} {'Cup Safe?':<10}")
    print("  " + "-" * 55)
    for point, label in test_points:
        h_cup = filter_cup.evaluate_safety(point)
        h_sponge = filter_sponge.evaluate_safety(point)
        safe = "YES" if h_cup > 0 else "NO"
        print(f"  {label:<20} {h_cup:<12.4f} {h_sponge:<12.4f} {safe:<10}")

    # --- Visualize CBF landscape ---
    print("\n[4/4] Generating visualization...")
    _visualize_latent_cbf(model, data, save_path="/home/claude/latent_cbf_landscape.png")

    # --- Run trajectory simulation ---
    print("\n  Running trajectory simulation with latent CBF filter...")
    from vlm_cbf_pipeline import ManipulationSimulator2D

    sim = ManipulationSimulator2D(dt=0.02)
    commands = sim.generate_figure_eight(
        center=np.array([0.05, 0.0]), radius=0.4, speed=0.2, n_steps=400
    )

    info_history = []
    for cmd in commands:
        u_cert, info = filter_cup.certify(sim.x_ee, cmd)
        sim.step(u_cert)
        info_history.append(info)

    n_mod = sum(1 for i in info_history if i["modified"])
    min_h = min(i["h_value"] for i in info_history)
    print(f"  Modified {n_mod}/{len(commands)} commands")
    print(f"  Minimum h(x) encountered: {min_h:.4f}")

    # Plot trajectory on CBF landscape
    _visualize_latent_cbf_with_trajectory(model, data, sim.trajectory,
                                          save_path="/home/claude/latent_cbf_trajectory.png")

    print("\n" + "=" * 70)
    print("Latent CBF demo complete!")
    print("Key files:")
    print("  - latent_cbf_landscape.png: Safety value landscape")
    print("  - latent_cbf_trajectory.png: Filtered trajectory on landscape")
    print("=" * 70)


def _visualize_latent_cbf(model, data, save_path: str):
    """Visualize the learned CBF landscape."""
    import matplotlib.pyplot as plt

    if not HAS_TORCH:
        return

    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_range = np.linspace(-0.6, 0.6, 150)
    y_range = np.linspace(-0.4, 0.4, 100)
    X, Y = np.meshgrid(x_range, y_range)

    scene_embed = torch.from_numpy(data["scene_embed"]).unsqueeze(0)

    for idx, (obj_name, obj_embed_key) in enumerate([("cup of water", "cup_embed"),
                                                       ("dry sponge", "sponge_embed")]):
        ax = axes[idx]
        obj_embed = torch.from_numpy(data[obj_embed_key]).unsqueeze(0)

        H = np.zeros_like(X)
        with torch.no_grad():
            for i in range(X.shape[0]):
                states = torch.from_numpy(
                    np.stack([X[i], Y[i]], axis=-1).astype(np.float32)
                )  # (W, 2)
                scene_batch = scene_embed.expand(states.shape[0], -1)
                obj_batch = obj_embed.expand(states.shape[0], -1)
                h_vals = model(scene_batch, obj_batch, states).squeeze(-1).numpy()
                H[i] = h_vals

        im = ax.contourf(X, Y, H, levels=50, cmap='RdYlGn', vmin=-2, vmax=3)
        ax.contour(X, Y, H, levels=[0], colors='black', linewidths=3)
        plt.colorbar(im, ax=ax, label='h(z) — safety margin')

        # Mark constraint regions
        from matplotlib.patches import Circle
        if "cup" in obj_name:
            for center, r in [([0.3, 0.0], 0.2), ([0.0, -0.2], 0.12)]:
                circle = Circle(center, r, fill=False, edgecolor='white',
                              linewidth=2, linestyle='--')
                ax.add_patch(circle)

        ax.set_title(f"Holding: {obj_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')

    plt.suptitle("Learned Latent CBF: Safety Landscape", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [VIZ] Saved to {save_path}")


def _visualize_latent_cbf_with_trajectory(model, data, trajectory, save_path: str):
    """Visualize trajectory on the learned CBF landscape."""
    import matplotlib.pyplot as plt

    if not HAS_TORCH:
        return

    model.eval()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    x_range = np.linspace(-0.6, 0.6, 150)
    y_range = np.linspace(-0.4, 0.4, 100)
    X, Y = np.meshgrid(x_range, y_range)

    scene_embed = torch.from_numpy(data["scene_embed"]).unsqueeze(0)
    obj_embed = torch.from_numpy(data["cup_embed"]).unsqueeze(0)

    H = np.zeros_like(X)
    with torch.no_grad():
        for i in range(X.shape[0]):
            states = torch.from_numpy(
                np.stack([X[i], Y[i]], axis=-1).astype(np.float32)
            )
            scene_batch = scene_embed.expand(states.shape[0], -1)
            obj_batch = obj_embed.expand(states.shape[0], -1)
            h_vals = model(scene_batch, obj_batch, states).squeeze(-1).numpy()
            H[i] = h_vals

    im = ax.contourf(X, Y, H, levels=50, cmap='RdYlGn', vmin=-2, vmax=3)
    ax.contour(X, Y, H, levels=[0], colors='black', linewidths=3, label='h=0 boundary')
    plt.colorbar(im, ax=ax, label='h(z) — safety margin')

    # Plot trajectory
    traj = np.array(trajectory)
    for i in range(len(traj) - 1):
        t_frac = i / max(len(traj) - 1, 1)
        color = plt.cm.plasma(t_frac)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, linewidth=2.5)

    ax.plot(traj[0, 0], traj[0, 1], 'wo', markersize=12, markeredgecolor='black', zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], 'ws', markersize=12, markeredgecolor='black', zorder=5)

    ax.set_title("Latent CBF-Filtered Trajectory (Cup of Water)", fontsize=13, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [VIZ] Saved to {save_path}")


if __name__ == "__main__":
    run_latent_cbf_demo()
