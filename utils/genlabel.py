#!/usr/bin/env python3
"""GenLabel relabeling for mixup, adapted to a learned embedding space.

Reference implementation and equations: Sohn, Shang, Chen, Moon,
Papailiopoulos and Lee, "GenLabel: Mixup Relabeling using Generative
Models", ICML 2022 (arXiv 2201.02354), and the authors' code at
github.com/UW-Madison-Lee-Lab/GenLabel_official. The relevant routines
there are estimate_gm and GenLabel_FullyConnected.fit in OpenML/train_test.py.

The original procedure, stated exactly:

  1. Fit one Gaussian per class by empirical mean and covariance, with the
     covariance floored as cov + tol * I, tol = 1e-4 (estimate_gm).
  2. For a mixed point x_mix, evaluate every class density and normalise
     across classes: t_gen = p_c(x_mix) / sum_k p_k(x_mix). Class priors do
     NOT enter the label, so it is a uniform-prior posterior. When the
     densities underflow to zero the authors fall back to a softmax over
     the log densities, which is the same quantity computed stably.
  3. Train against gamma * CE(f(x_mix), t_gen)
                   + (1 - gamma) * CE(f(x_mix), lam * y_a + (1 - lam) * y_b),
     with gamma chosen by 6-fold cross-validation over
     {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} (Appendix E, confirmed against the
     paper 2026-10-14). Pinning gamma at 1.0, as every arm here did before
     queue 118, is the single point on that grid at which a saturated
     generative label reaches the loss undiluted.

Cross entropy against a soft target is linear in the target, so the two
weighted CE terms are identical to a single CE against the blended target
gamma * t_gen + (1 - gamma) * t_vanilla. This module builds that blended
target and hands it to the same soft_target_loss every other arm uses,
which keeps the loss path byte-identical across arms.

FOUR ADAPTATIONS, each forced by this setting rather than chosen.

  CHANGING EMBEDDINGS. The authors give two latent-space variants, checked
  against the paper on 2026-10-14. Algorithm 2 (Appendix A.1) trains the
  classifier first, fits the densities on the resulting features, then
  FREEZES the feature extractor and fine-tunes only the classification
  head. Algorithm 3 (Appendix A.2), the one they actually use for the image
  datasets, instead re-estimates the class-conditional mean and covariance
  AT EVERY BATCH while the network keeps training. Freezing our backbone
  would break the requirement that backbone and optimisation match across
  arms, so we follow the second variant but refit at every epoch boundary
  rather than every batch, from sufficient statistics accumulated during
  that epoch. That is between their two settings, not outside them. The statistics cost one
  [D, B] @ [B, D] matmul per batch and no extra forward passes. The target
  is therefore built from densities that are one epoch stale, and is a
  constant with respect to the model: the mixed point is detached before
  scoring, exactly as the authors detach into numpy.

  WARMUP. No density exists before the first refit, which happens at the
  end of epoch 1, so epoch 1 trains on the vanilla mixup target and is
  identical to the vanilla mixup control. This mirrors the authors' warmup
  phase, at the shortest length their recipe admits.

  DIMENSIONALITY, AND THE AUTHORS' OWN SCOPE. This is the adaptation that
  matters most, and the paper is explicit about the limit. They evaluate on
  OpenML datasets filtered to FEWER THAN 20 FEATURES and fewer than 5000
  samples, and they then report only the 82 of 133 datasets on which a KDE
  or Gaussian mixture reaches above 95 percent train accuracy, discarding
  the rest. Their footnote states that the algorithm "will not work as
  expected if the chosen generative model has a poor fit to the data", and
  Section 6.3 reports that on high-dimensional image data "GenLabel has a
  marginal gain".

  Our post-fusion embedding is 1024-d with roughly 15k training samples per
  class, which is outside that stated regime in both respects. A per-class
  full covariance is badly conditioned, and a 1024-d Gaussian log density
  produces class score gaps of thousands of nats, so the normalised label
  is one-hot on essentially every blend and GenLabel silently becomes a
  hard relabeling scheme. Measured on synthetic 1024-d data with four
  classes: mean top posterior 0.9989, 98.7 percent of rows above 0.999,
  mean target entropy 0.003.

  An optional projection to genlabel_pca_dim, fitted from the same pooled
  accumulator and so costing no extra pass, keeps the estimate well posed
  and puts the method back inside the dimensionality its authors tested.
  Set genlabel_pca_dim to 0 for the unreduced form, which is faithful to
  the equations while being outside the regime the authors claim.

  COVARIANCE REGULARISATION. The authors' cov + tol * I floor is kept and
  configurable, and a Ledoit-Wolf style shrinkage toward a scaled identity,
  (1 - rho) * S + rho * (trace(S) / d) * I, is available on top of it.
  Both are tuned on development data, never on test.
"""

from typing import Optional, Tuple

import torch

from utils.muted_mixup import sample_lambda


class EmbeddingClassDensity:
    """Class-conditional Gaussians over a moving embedding space.

    Sufficient statistics are accumulated during an epoch and converted to
    means and inverse covariances at the epoch boundary, so the densities
    track the representation without any extra forward pass.

    Attributes:
        ready: True once at least one refit has produced usable statistics.
    """

    def __init__(self, num_classes: int, dim: int, device: torch.device,
                 pca_dim: int = 0, shrinkage: float = 0.0,
                 tol: float = 1e-4, proj_mode: str = "within") -> None:
        """Create an empty density bank.

        Args:
            num_classes: number of emotion classes.
            dim: embedding dimensionality.
            device: device the accumulators live on.
            pca_dim: project to this many dimensions before fitting; 0 keeps
                the full space.
            shrinkage: rho in (1 - rho) * S + rho * (trace(S) / d) * I.
            tol: additive diagonal floor, the authors' cov + tol * I.
            proj_mode: how the projection basis is chosen when pca_dim is
                positive. "within" keeps the top eigenvectors of the pooled
                within-class covariance, which is what the PCA-64 arm has
                always used. "lda" solves the generalised eigenproblem
                S_b w = e * S_w w and keeps the most class-discriminative
                directions instead.

                The distinction is cosmetic at 64 dimensions out of 1024 and
                decisive at 3. Top within-class directions are the directions
                of greatest spread INSIDE each class, so at 3 dimensions they
                retain mostly nuisance variance: a measured mean top
                posterior of 0.36 against 0.82 at 64 dimensions, which is
                close to the uniform 0.25. A 3-dimensional arm built that way
                would lose for want of class information rather than for want
                of affect information, which is not the comparison the
                space-matched control is supposed to make. LDA returns at
                most num_classes - 1 directions, which for four classes is
                exactly the 3 of VAD.

        Raises:
            ValueError: if proj_mode is unknown, or if "lda" is asked for
                more than num_classes - 1 dimensions.
        """
        if proj_mode not in ("within", "lda"):
            raise ValueError(
                "proj_mode must be 'within' or 'lda', got %r" % (proj_mode,))
        if proj_mode == "lda" and int(pca_dim) > int(num_classes) - 1:
            raise ValueError(
                "LDA yields at most num_classes - 1 = %d directions, but "
                "pca_dim is %d. Reduce pca_dim or use proj_mode 'within'."
                % (int(num_classes) - 1, int(pca_dim)))
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.device = device
        self.pca_dim = int(pca_dim)
        self.shrinkage = float(shrinkage)
        self.tol = float(tol)
        self.proj_mode = str(proj_mode)
        self.ready = False
        self.proj: Optional[torch.Tensor] = None
        self.means: Optional[torch.Tensor] = None
        self.inv_covs: Optional[torch.Tensor] = None
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Zero the per-class count, first and second moment accumulators."""
        c, d = self.num_classes, self.dim
        self.n = torch.zeros(c, dtype=torch.float64, device=self.device)
        self.s1 = torch.zeros(c, d, dtype=torch.float64, device=self.device)
        self.s2 = torch.zeros(c, d, d, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def accumulate(self, embeddings: torch.Tensor,
                   labels: torch.Tensor) -> None:
        """Add one batch to the running per-class moments.

        Args:
            embeddings: [B, D] post-fusion embeddings.
            labels: [B] integer class labels.
        """
        z = embeddings.detach().to(torch.float64)
        y = labels.long()
        for c in range(self.num_classes):
            m = y == c
            if not bool(m.any()):
                continue
            zc = z[m]
            self.n[c] += zc.shape[0]
            self.s1[c] += zc.sum(dim=0)
            self.s2[c] += zc.t() @ zc

    @torch.no_grad()
    def _lda_projection(self, means: torch.Tensor,
                        pooled: torch.Tensor) -> Optional[torch.Tensor]:
        """Most class-discriminative directions, by multi-class LDA.

        Solves the generalised eigenproblem S_b w = e * S_w w by whitening
        with S_w, which is numerically better behaved than inverting S_w and
        eigendecomposing a non-symmetric product. S_w is floored by self.tol
        before the Cholesky because a 1024-d within-class covariance
        estimated from one epoch is not guaranteed positive definite.

        Args:
            means: [C, D] per-class means in the unprojected space.
            pooled: [D, D] symmetric pooled within-class covariance, S_w.

        Returns:
            [D, pca_dim] projection whose columns are the leading LDA
            directions, or None if the factorisation failed, in which case
            the caller treats this epoch as having no usable density.
        """
        d = pooled.shape[0]
        eye = torch.eye(d, dtype=pooled.dtype, device=pooled.device)
        s_w = pooled + self.tol * eye

        # Between-class scatter, weighted by class frequency.
        weights = (self.n / self.n.sum()).unsqueeze(1)
        grand = (means * weights).sum(dim=0, keepdim=True)
        delta = means - grand
        s_b = (delta * weights).t() @ delta
        s_b = 0.5 * (s_b + s_b.t())

        try:
            chol = torch.linalg.cholesky(s_w)
            # whitened = L^-1 S_b L^-T, symmetric by construction.
            tmp = torch.linalg.solve_triangular(chol, s_b, upper=False)
            whitened = torch.linalg.solve_triangular(
                chol, tmp.t(), upper=False).t()
            whitened = 0.5 * (whitened + whitened.t())
            _, evecs = torch.linalg.eigh(whitened)
            # Map back out of the whitened space: w = L^-T v.
            basis = torch.linalg.solve_triangular(
                chol.t(), evecs[:, -self.pca_dim:], upper=True)
        except RuntimeError:
            return None

        if not bool(torch.isfinite(basis).all()):
            return None
        # Unit columns so the projected scale does not depend on S_w's
        # conditioning, which varies across epochs as the backbone moves.
        norms = basis.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return (basis / norms).contiguous()

    @torch.no_grad()
    def refit(self) -> bool:
        """Convert accumulated moments into means and inverse covariances.

        The accumulators are cleared afterwards, so each refit describes the
        epoch that just ended rather than all of training.

        Returns:
            True if every class had enough samples and the fit succeeded.
        """
        floor = self.dim if self.pca_dim <= 0 else self.pca_dim
        if bool((self.n <= floor + 1).any()):
            self._reset_accumulators()
            return False

        means = self.s1 / self.n.unsqueeze(1)
        # Per-class scatter, unbiased.
        covs = []
        for c in range(self.num_classes):
            mu = means[c].unsqueeze(1)
            sc = (self.s2[c] - self.n[c] * (mu @ mu.t())) / (self.n[c] - 1.0)
            covs.append(sc)
        covs = torch.stack(covs, dim=0)

        if self.pca_dim > 0 and self.pca_dim < self.dim:
            # Pooled within-class covariance, fitted from the same
            # accumulators, so neither projection costs an extra pass.
            pooled = (covs * self.n.view(-1, 1, 1)).sum(0) / self.n.sum()
            pooled = 0.5 * (pooled + pooled.t())
            if self.proj_mode == "lda":
                self.proj = self._lda_projection(means, pooled)
            else:
                # Top eigenvectors of the within-class covariance. These are
                # the directions of greatest spread inside each class, which
                # is not the same as the directions that separate them; see
                # proj_mode in __init__ for why that matters below about 64
                # dimensions.
                _, evecs = torch.linalg.eigh(pooled)
                self.proj = evecs[:, -self.pca_dim:].contiguous()
            if self.proj is None:
                self._reset_accumulators()
                return False
            means = means @ self.proj
            covs = torch.einsum('di,cde,ej->cij', self.proj, covs, self.proj)
        else:
            self.proj = None

        d = covs.shape[-1]
        eye = torch.eye(d, dtype=torch.float64, device=self.device)
        if self.shrinkage > 0.0:
            tr = torch.diagonal(covs, dim1=-2, dim2=-1).sum(-1) / d
            covs = ((1.0 - self.shrinkage) * covs
                    + self.shrinkage * tr.view(-1, 1, 1) * eye.unsqueeze(0))
        covs = covs + self.tol * eye.unsqueeze(0)

        try:
            inv = torch.linalg.inv(covs)
        except RuntimeError:
            self._reset_accumulators()
            return False

        self.means = means.to(torch.float32)
        self.inv_covs = (0.5 * (inv + inv.transpose(-1, -2))).to(torch.float32)
        self.ready = True
        self._reset_accumulators()
        return True

    @torch.no_grad()
    def posterior(self, points: torch.Tensor) -> torch.Tensor:
        """Uniform-prior class posterior at each point.

        Computed as a softmax over log densities, which is the authors'
        stable fallback and is identical to normalising the densities.

        Args:
            points: [N, D] embeddings in the original embedding space.

        Returns:
            [N, C] posterior, rows summing to one.
        """
        z = points.detach().to(torch.float32)
        if self.proj is not None:
            z = z @ self.proj.to(torch.float32)
        scores = []
        for c in range(self.num_classes):
            delta = z - self.means[c].unsqueeze(0)
            quad = torch.einsum('ni,ij,nj->n', delta, self.inv_covs[c], delta)
            _, logdet = torch.linalg.slogdet(self.inv_covs[c].double())
            scores.append(-0.5 * quad + 0.5 * logdet.to(torch.float32))
        return torch.softmax(torch.stack(scores, dim=1), dim=1)


def genlabel_mixup_step(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_layer: torch.nn.Module,
    density: EmbeddingClassDensity,
    num_classes: int,
    alpha: float = 2.0,
    gamma: float = 1.0,
    fold_lambda: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """One GenLabel mixup step in embedding space.

    Pair selection, the lambda distribution and the blend are identical to
    general_soft_mixup_step, so the only difference against the VAD-informed
    arm is where the target comes from: class-conditional densities fitted
    on training EMBEDDINGS here, on training VAD annotations there.

    Args:
        embeddings: [B, hidden_dim] post-fusion embeddings.
        labels: [B] class labels.
        output_layer: maps embeddings to logits.
        density: fitted class-conditional density bank.
        num_classes: number of classes.
        alpha: Beta(alpha, alpha) parameter for the mixing coefficient.
        gamma: weight on the GenLabel target against the vanilla mixup
            target, the authors' cross-validated hyperparameter. 1.0 is a
            pure GenLabel target.
        fold_lambda: fold the coefficient into [0.5, 1], matching the arm
            under test.
        generator: optional RNG.

    Returns:
        (logits, target, lam), target [M, C] soft. (None, None, None) for a
        batch of one.
    """
    device = embeddings.device
    n = embeddings.size(0)
    if n < 2:
        return None, None, None

    pick = torch.randint(n, (n,), device=device, generator=generator)
    clash = pick == torch.arange(n, device=device)
    if clash.any():
        pick[clash] = (pick[clash] + 1) % n

    if fold_lambda:
        lam = sample_lambda(n, alpha, device)
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device)
    lam = lam.unsqueeze(1)
    mixed = lam * embeddings + (1.0 - lam) * embeddings[pick]
    logits = output_layer(mixed)

    eye = torch.eye(num_classes, device=device, dtype=logits.dtype)
    vanilla = lam * eye[labels.long()] + (1.0 - lam) * eye[labels[pick].long()]

    if not density.ready:
        # Warmup: no density yet, so this epoch is the vanilla control.
        return logits, vanilla, lam.squeeze(1)

    gen = density.posterior(mixed).to(logits.dtype)
    target = gamma * gen + (1.0 - gamma) * vanilla
    return logits, target, lam.squeeze(1)
