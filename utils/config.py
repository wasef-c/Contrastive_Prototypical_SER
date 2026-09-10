#!/usr/bin/env python3
"""
Configuration system for emotion recognition with contrastive learning
"""

import yaml
from utils.prototypicality import DEFAULT_EXPECTED_VAD


# Encoders that consume raw waveforms rather than precomputed feature vectors.
# Kept in one place because the membership test is needed in the dataset
# loader, the trainer and the runner; when it was written out by hand at each
# site, adding an encoder silently loaded the precomputed-feature datasets
# instead, whose label columns differ, and every row was rejected as invalid.
RAW_AUDIO_ENCODERS = ("wav2vec2", "emotion2vec", "wavlm")


class Config:
    """Clean configuration for contrastive learning experiments"""

    def __init__(self, **kwargs):
        # Dataset
        self.train_dataset = "MSPP"
        self.test_datasets = []  # Empty = all others
        self.val_split = 0.1

        # Model architecture
        self.modality = "both"  # "audio", "text", "both"
        self.audio_dim = 768
        self.text_model_name = "bert-base-uncased"
        self.text_max_length = 128
        self.hidden_dim = 1024
        self.num_classes = 4

        # BERT unfreezing
        self.unfreeze_bert_layers = 0  # 0 = fully frozen, 2-4 = unfreeze top N layers
        self.bert_learning_rate = 5e-7  # 10x lower than main LR

        # Audio encoder
        # "preextracted", "wav2vec2", "emotion2vec", or "wavlm"
        self.audio_encoder_type = "preextracted"
        self.audio_model_name = "facebook/wav2vec2-base-960h"
        self.unfreeze_audio_layers = 0  # 0 = fully frozen, 2-4 = unfreeze top N layers
        self.audio_learning_rate = 5e-7  # differential LR for wav2vec2/emotion2vec
        # Temporal pooling of emotion2vec frame features. "mean" (768)
        # discards every within-utterance dynamic: a happy utterance that
        # builds from calm and one that is steadily warm have nearly the
        # same mean. "mean_std" (1536) adds how much the signal moves,
        # "mean_std_halves" (3072) adds direction of change. All stay
        # cacheable, so training speed is unchanged.
        # "mean" (768), "mean_std" (1536), "mean_std_halves" (3072),
        # "seg<K>" (K segment means, K*768) or "dup<K>" (the mean repeated
        # K times). dup<K> is the capacity control: same width and same
        # downstream parameter count as a real temporal mode, but zero new
        # information, so it separates "temporal detail helps" from "a
        # wider fusion layer helps".
        self.audio_pooling = "mean"
        # Frame-level mode ("frames") keeps the time axis and lets a
        # learned module pool it. num_frames sets the fixed per-utterance
        # frame count in the memmapped cache; num_attn_heads_pool sets how
        # many independent queries attend over those frames.
        # use_mean_pool_control swaps AttentionPool for a capacity-matched
        # placebo that just repeats the mean, which is the only way to tell
        # attending-over-time from a wider fusion layer.
        self.num_frames = 32
        self.num_attn_heads_pool = 4
        self.attn_pool_hidden = 256
        self.use_mean_pool_control = False
        self.max_audio_seconds = 40  # truncate waveforms to this length (caps VRAM usage)
        self.emotion2vec_upstream_dir = "/home/rml/Documents/pythontest/emotion2vec/upstream"

        # Fusion (for multimodal)
        self.fusion_type = "cross_attention"
        self.fusion_hidden_dim = 512
        self.num_attention_heads = 8

        # Task
        self.task_type = "classification"  # "classification" or "regression"
        self.vad_output_dim = 3

        # Collapse the target to neutral (0) vs emotional (1). Used for the
        # presence-detector member of the stacked ensemble, where the second
        # member still predicts all four classes. Set num_classes to 2 with it.
        self.binary_neutral = False

        # Drop neutral from the split entirely and renumber the emotional
        # classes to 0..K-1. The second stage of the hierarchical system,
        # where a detector already owns the presence decision. Set
        # num_classes to 3 with it. Mutually exclusive with binary_neutral.
        self.drop_neutral = False

        # Feature-level fusion of a neutral-vs-emotional presence branch into
        # the four-way head. Unlike the stacked ensemble, which combines two
        # trained models at the decision level, this shares one backbone and
        # concatenates the detector's hidden representation into the
        # classification head, so presence evidence reaches the classifier
        # before it is collapsed into two probabilities.
        self.use_detector_fusion = False
        self.detector_fusion_dim = 128
        self.detector_fusion_weight = 0.5
        # Placebo: keep the branch, its parameters and the concatenation, but
        # train its head on shuffled presence targets so it carries no
        # presence information. Any gain that survives this is attributable to
        # the detector signal rather than the extra capacity.
        self.detector_fusion_shuffle = False

        # Dump per-sample logits for the held-out split and every test corpus
        # at the end of a run, so ensembling and threshold sweeps are offline
        # work rather than fresh GPU runs.
        self.save_predictions = False

        # Training
        self.num_epochs = 60
        self.batch_size = 32
        self.gradient_accumulation_steps = 1  # set >1 with small batch_size to simulate larger effective batch
        self.learning_rate = 5e-6
        self.weight_decay = 5e-6
        self.dropout = 0.1
        self.eval_batch_size = None  # defaults to batch_size if not set

        # Prototypicality
        self.expected_vad = DEFAULT_EXPECTED_VAD.copy()

        # Contrastive Learning
        self.use_contrastive = False
        self.contrastive_loss_type = "supervised"  # "supervised", "prototypical_v1", "prototypical_v2", "prototypical_v3"
        self.contrastive_weight = 0.5
        self.contrastive_temperature = 0.07
        self.contrastive_warmup_epochs = 5

        # Projection head (for contrastive learning)
        self.projection_dim = 128
        self.projection_hidden_dim = 512

        # Prototypicality weighting
        self.prototypical_alpha = 1.0  # For v1: exp(-alpha * difficulty)
        self.prototypical_beta = 0.5   # For v2: pair-level weighting
        self.prototypical_threshold = 1.0  # For v3: binary threshold

        # Prototype-anchored loss params
        self.margin_base = 0.1  # Min margin for prototypical samples
        self.separation_margin = 2.0  # Min distance between class prototypes
        self.separation_weight = 0.5  # Weight for prototype separation loss
        self.alignment_weight = 1.0  # Weight for cross-domain alignment (multiDS)

        # Multi-view prototypicality
        self.use_multiview_prototypicality = False
        self.mv_weight_vad = 0.4    # Weight for VAD distance view
        self.mv_weight_cross = 0.3  # Weight for cross-modal agreement view
        self.mv_weight_embed = 0.3  # Weight for embedding-space distance view

        # Hard negative mining
        self.use_hard_negative_mining = False
        self.hard_negative_weight = 0.3

        # Prototype memory bank
        self.use_memory_bank = False
        self.bank_size = 64
        self.bank_momentum = 0.5
        self.bank_threshold = 0.5

        # Cross-modal projection
        self.use_cross_modal_projection = False
        self.cross_modal_dim = 256
        self.cross_modal_weight = 0.1

        # Salience-weighted primary loss. Unlike prototypical weighting
        # below, difficulty is the signed displacement toward the neutral
        # region rather than the unsigned distance to the class prototype.
        # The unsigned version does not predict this model's errors; see
        # motivation/README.md.
        self.use_salience_weighting = False
        self.salience_beta = 0.5        # exp(beta * standardised difficulty)
        self.salience_clip = 3.0        # max ratio between largest and smallest weight
        self.salience_shuffle = False   # placebo: same weight distribution, no link to sample
        self.salience_scope = "both"   # "both" | "emotional" | "neutral"
        self.salience_gate_lr = 1e-2   # gate scalars need their own LR to move at all

        # Muted-emotion mixup: synthesise the sparse low-salience
        # emotional region instead of reweighting it. See utils/muted_mixup.py.
        self.use_muted_mixup = False
        # Calibrate the blend coefficient from each sample's measured VAD
        # intensity instead of drawing it from Beta, so a loud utterance is
        # muted hard and a quiet one is left alone, and the synthetic samples
        # land in the low-intensity region the training set barely covers.
        self.muted_mixup_vad_calibrated = False
        # Upper bound of the target intensity band, as a quantile of the
        # batch's own intensity spread. Lower pushes further into the quiet end.
        self.muted_mixup_vad_quantile = 0.5
        # Control for the calibrated variant: permute the coefficients across
        # samples, matching their distribution while destroying the link
        # between a sample's real intensity and how far it gets muted.
        self.muted_mixup_vad_shuffle = False
        # Move each class toward whichever intensity end its own training data
        # is thin at, instead of muting every class toward neutral. On
        # MSP-Podcast only angry is sparse when quiet; sad is already
        # concentrated there, so muting it adds density at the neutral
        # boundary rather than filling a gap.
        # Bidirectional blending gated by measured VAD: only emotional
        # samples that are loud for their class get muted, and only clearly
        # central neutral samples take on a little emotion. Both sides gain
        # samples in proportion, so the prior is left where it was.
        self.muted_mixup_gated_symmetric = False
        self.muted_mixup_gate_quantile = 0.5
        # Which modality the blend touches. Intensity is prosodic, so
        # audio-only is the version that states the hypothesis; text-only is
        # its mirror-image control and both is the original behaviour.
        self.muted_mixup_mix_audio = True
        self.muted_mixup_mix_text = True
        self.muted_mixup_sparse_fill = False
        # Which classes participate. Empty means every emotional class. Use
        # e.g. [3] to run the angry-only variant, the one case where the
        # mechanism provably reaches an empty region rather than interpolating.
        self.muted_mixup_sparse_classes = []
        # Where in the sparse tail to aim, as a quantile of that class's own
        # intensity distribution.
        self.muted_mixup_sparse_target_q = 0.1
        self.muted_mixup_alpha = 2.0
        self.muted_mixup_weight = 0.5
        self.muted_mixup_control = False   # placebo: blend toward another emotion
        self.muted_mixup_symmetric = False  # blend both ways to hold the prior fixed

        # Prototypicality-weighted primary loss
        self.use_prototypical_weighting = False
        self.prototypical_weighting_alpha = 2.0  # exp(-alpha * difficulty)
        # Source of difficulty for CE weighting: "vad" | "agreement" | "both"
        self.ce_weight_source = "vad"
        self.ce_weight_both_vad_w = 0.5  # blend weight for "both" (agreement gets 1-w)
        # If True, up-weight atypical samples: exp(+alpha*diff) instead of exp(-alpha*diff)
        self.ce_weight_invert = False
        # Learnable VAD centroids
        self.use_learned_centroids = False
        self.learned_centroid_mode = "ema"  # "ema" | "grad"
        self.learned_centroid_momentum = 0.9  # EMA momentum (higher = slower update)
        self.learned_centroid_lr = 1e-3  # LR for grad-mode centroid params
        self.use_prototypical_label_smoothing = False
        self.label_smoothing_beta = 0.5  # smoothing = beta * difficulty
        self.label_smoothing_max = 0.6  # cap smoothing to avoid total flattening

        # Auxiliary prototypicality prediction
        self.use_proto_predictor = False
        # Control task for the prototypicality head: permute the targets
        # within the batch so the head keeps its capacity but predicts a
        # value that does not belong to the sample.
        self.proto_predictor_shuffle = False
        # Linear ramp of the auxiliary weight over the first N epochs.
        self.proto_predictor_warmup_epochs = 0
        # Decay the auxiliary weight to zero over the final N epochs.
        self.proto_predictor_anneal_epochs = 0
        # Log cos(g_main, g_aux) on the shared trunk every N steps. Zero
        # disables. This is the diagnostic for whether the auxiliary
        # gradient still points the same way as the classifier once the
        # encoders are unfrozen; near-zero cosine means the aux is acting
        # as structured gradient noise rather than transferring anything.
        self.proto_grad_cosine_every = 0

        # Group similar-duration utterances into the same batch. Only
        # applies to raw-audio (unfrozen encoder) runs, where the collate
        # pads to the batch maximum and random batching wastes ~40% of
        # compute on padding.
        self.use_length_bucketing = False
        self.length_bucket_window = 50
        # What the prototypicality head predicts. 'scalar' collapses the
        # 3-D position into one number, so loud-atypical and quiet-atypical
        # share a target; 'residual' keeps the direction; 'subtype' asks
        # which cluster within the class, which is the version that does
        # not assume each emotion is unimodal.
        # Couple the mixup and the prototypicality head: a blended sample
        # keeps its hard class label but gets an auxiliary target computed
        # from where the blend landed in VAD space. Requires
        # use_proto_predictor and proto_target='residual'.
        self.use_consistent_mixup = False
        self.consistent_mixup_weight = 0.5
        # scalar | residual | subtype | allclass | signed.
        # allclass predicts the residual to EVERY class centre, which is
        # what the angry/sad boundary needs; signed splits the residual
        # into a signed component along the class's own direction away
        # from neutral and the leftover, so intensifying within a class
        # is no longer scored as atypical.
        self.proto_target = 'scalar'
        # Euclidean treats V, A and D as equally scaled and uncorrelated.
        # Measured per-class covariances have condition numbers of 6 to 9,
        # so that is wrong; mahalanobis corrects it, and combined with
        # 'residual' gives a whitened residual whose norm is that distance.
        self.proto_metric = 'euclidean'  # euclidean | mahalanobis
        self.proto_predictor_weight = 0.5  # λ for MSE(pred_proto, actual_proto)
        self.proto_predictor_hidden_dim = 256

        # Domain adversarial training
        self.use_domain_adversarial = False
        self.adversarial_weight = 0.1  # Weight for adversarial loss
        self.adversarial_alpha = 2.0  # Prototypicality weight decay for adversarial
        self.adversarial_hidden_dim = 256  # Hidden dim of domain discriminator
        self.use_prototypical_adversarial = True  # Weight adversarial by prototypicality

        # Modality-level domain adversarial (applies GRL to audio/text pre-fusion)
        self.use_modality_adversarial = False
        self.modality_adv_weight = 0.1  # Weight for per-modality adversarial loss

        # Adversarial GRL scheduling
        self.adversarial_peak_lambda = 1.0  # Peak lambda value after warmup
        self.adversarial_warmup_frac = 0.5  # Fraction of training used to ramp lambda to peak

        # VADmix: cross-corpus feature mixup with VAD-derived soft targets
        self.use_vadmix = False
        self.vadmix_alpha = 0.2  # Beta(alpha, alpha) mixing parameter
        self.vadmix_weight = 1.0  # Multiplier on VADmix soft CE in combined loss
        self.vadmix_temperature = 0.5  # Softmax temperature for VAD->soft-label mapping
        self.vadmix_cross_corpus_only = True  # Pair samples across corpora when possible

        # VREx: Variance Risk Extrapolation. Treats corpora as IRM environments
        # and penalizes the variance of per-corpus mean losses. No VAD required.
        self.use_vrex = False
        self.vrex_lambda = 10.0  # Peak coefficient on variance penalty after warmup
        self.vrex_warmup_frac = 0.3  # Fraction of training used to ramp lambda from 0 -> peak

        # Corpus-balanced sampling: WeightedRandomSampler that gives each corpus
        # equal probability mass per batch (overrides natural corpus size ratios).
        # Required for VREx so per-corpus loss estimates have enough samples.
        self.use_corpus_balanced_sampler = False
        # Optional explicit per-corpus oversampling weights. None -> uniform across corpora.
        # Example: {"MSPP": 1.0, "IEMO": 4.0} oversamples IEMO 4x relative to MSPP weight.
        self.corpus_sample_weights = None

        # Two-stage prototypicality training
        self.use_two_stage_training = False
        self.stage1_percentile = 50  # bottom N% most prototypical for stage 1
        self.stage1_epochs = 30     # epochs for stage 1
        self.stage2_lr_factor = 0.1  # reduce LR by this factor for stage 2

        # Curriculum learning: gradually introduce samples over the first
        # curriculum_epochs epochs, ordered by a difficulty signal.
        #   type: "difficulty" easiest first (VAD-distance based, ascending)
        #         "inverse_difficulty" hardest first
        #         "class_balance" start with class 0, add others as pacing grows
        #         "random"       random subset of the pacing fraction
        #         "none"         all data from epoch 0 (disables subsetting)
        #         "preset_order" sort by ascending difficulty once, then take the head
        #   pacing: "linear" | "sqrt" | "log" fraction schedule over curriculum_epochs
        # After curriculum_epochs, all data is used and dropout is bumped to
        # post_curriculum_dropout to compensate for the harder examples arriving.
        self.use_curriculum_learning = False
        self.curriculum_epochs = 10
        self.curriculum_pacing = "linear"
        self.curriculum_type = "difficulty"
        self.post_curriculum_dropout = 0.6

        # Prototypical/Atypical sub-label split. When on, the classifier head is
        # widened to 2 * num_classes and each training sample gets a sub-label
        # sub = 2 * class + is_atypical. is_atypical is precomputed at init from
        # the sample's VAD distance to the class center. At eval, the 2*C logits
        # are collapsed back to C class probs via softmax pair sum, so test
        # corpora never need VAD.
        #   split_criterion:
        #     "per_class_median" -> median distance within each class (50/50 per class)
        #     "global_median"    -> single median across all training samples
        #   center_source:
        #     "theoretical" -> centroids from config.expected_vad (yaml)
        #     "class_means" -> empirical per-class VAD means from the train set
        #     "learnable"   -> reserved; not implemented in this pass
        # Auxiliary VAD-cluster multi-task. Fits a k-means on the VAD points of
        # the VAD-annotated training samples, snapshots the k centroids, and
        # adds a lightweight aux head on top of the fused embedding that
        # predicts each sample's nearest cluster ID (CE, masked to samples
        # that have real VAD). Total loss = classification + weight * aux.
        # At eval the aux head is unused, so test corpora do not need VAD.
        # Clustering is across all training corpora combined, so cross-corpus
        # training induces a shared cross-corpus target.
        self.use_aux_vad_cluster = False
        self.aux_vad_cluster_k = 8
        self.aux_vad_cluster_weight = 0.2
        # Aux head architecture. 1 = single Linear(hidden_dim, k). 2 = a
        # two-layer MLP (Linear -> ReLU -> Linear) with a hidden width equal
        # to hidden_dim // 2. Depth > 1 lets the head absorb some capacity
        # so gradient pressure on the encoder is milder; useful when the
        # aux weight is high.
        self.aux_vad_head_depth = 1
        # KMeans init strategy for the aux VAD clusters.
        #   "random"            -> sklearn kmeans++ (arbitrary but reproducible)
        #   "class_prototypes"  -> seed each class-pair with (class mean,
        #                          atypical exemplar) so the k clusters have a
        #                          semantic lineage to (class, mode). Forces
        #                          k = 2 * num_classes; k-means still refines
        #                          the seeds against the data.
        #   "random_partition"  -> k centroids sampled uniformly in the VAD
        #                          bounding box, never fitted. An arbitrary
        #                          Voronoi partition that still carries VAD
        #                          info; placebo control for "does the
        #                          partition geometry matter".
        self.aux_vad_cluster_init = "random"
        # Clustering scope.
        #   "global"    -> one k-means over all VAD points pooled across
        #                  classes (aux_vad_cluster_k cells total). Cells may
        #                  straddle class boundaries.
        #   "per_class" -> a separate k-means inside each class, giving
        #                  happy-1..happy-n, angry-1..angry-n, etc. The aux
        #                  label space becomes
        #                  num_classes * aux_vad_clusters_per_class and the
        #                  aux task is a strict refinement of the primary
        #                  label. aux_vad_cluster_k is ignored in this mode.
        self.aux_vad_cluster_scope = "global"
        self.aux_vad_clusters_per_class = 2
        # Aux task variant for the mechanism ablation.
        #   "cluster"    -> predict nearest-centroid cluster ID (CE). Default.
        #   "regression" -> predict raw (V, A, D) with MSE; no centroids.
        #                   Tests whether continuous VAD supervision matches
        #                   the discretized cluster target.
        #   "subtype"    -> predict the annotator-supplied secondary tags
        #                   (multi-label BCE). Targets come from the joined
        #                   MSP-Podcast annotator metadata, not from k-means,
        #                   so the subtypes are the ones humans named rather
        #                   than ones inferred from VAD geometry.
        self.aux_vad_task = "cluster"
        # Clusters for aux_vad_task="resid_cluster". Under scope "shared"
        # this is the total number of deviation modes across all classes;
        # under "per_class" it is the number fitted inside each class.
        # Merge MSP-Podcast's official Train and Development splits into one
        # training pool. Development holds 46 percent of the corpus's angry
        # data, and every evaluation corpus here is external, so holding it
        # out costs minority-class coverage and buys nothing.
        # Nearest-prototype distance stream mixed into the class logits.
        # Requires aux_vad_task="regression" so predicted VAD is available
        # without a label at test time. The residual signal has repeatedly
        # shown up in AUC but not UAR, meaning it shapes the representation
        # without reaching the argmax; this puts it in the decision.
        # Fit the class prototypes from only the highest-agreement samples
        # in each class, at this within-class quantile. 0 disables it and
        # uses every sample, which is the historical behaviour. A contested
        # utterance's consensus VAD summarises a disagreement rather than
        # measuring the emotion, so it does not belong in the definition of
        # what that emotion sounds like.
        # Class-stratified subsample of the training pool, 1.0 = all rows.
        # Exists for unfrozen sweeps, where a full-pool epoch costs 47 minutes
        # because every step runs raw audio through emotion2vec. Validation is
        # never subsampled.
        self.train_subsample_frac = 1.0
        # Weight the prototypicality-head loss by annotator agreement.
        # "per_class" whitens each class's residual by its own covariance,
        # which puts every class in a different coordinate system; "shared"
        # pools the within-class scatter so all residuals live in one space.
        # VAD axes used by the prototypicality target: 0=valence,
        # 1=arousal, 2=dominance. [0,1] drops dominance.
        self.proto_vad_dims = [0, 1, 2]
        self.proto_whiten_scope = "per_class"
        self.proto_agreement_weight = False
        # Which agreement notion the centroid filter selects on:
        # "class" = fraction of annotators choosing the majority label,
        # "vad"   = inverse mean annotator VAD dispersion.
        self.proto_consensus_source = "class"
        self.proto_centroid_consensus_q = 0.0
        self.use_proto_dist_logits = False
        # Floor on the prototype-distance mixing weight. 0 leaves it free,
        # which measured at 0.003: given the choice the optimiser ignores the
        # stream. A positive floor forces a fixed minimum contribution, so
        # the arm tests whether being held to the prototype geometry helps
        # rather than whether the model elects to use it.
        self.proto_dist_alpha_min = 0.0
        # Permuted-prototype control for the stream above.
        self.proto_dist_permute = False
        self.mspp_merge_splits = True
        self.aux_resid_cluster_k = 6
        # Number of secondary tags in the subtype target vector. Must match
        # the width of subtype_dist in the joined metadata.
        self.aux_vad_subtype_dim = 16
        # Join the per-annotator statistics onto the training corpus. Required
        # by aux_vad_task="subtype" and by aux_vad_agreement_weight.
        self.use_annotator_meta = False
        # Scale each sample's auxiliary loss by 1 / (1 + mean annotator VAD
        # std), so the head spends capacity on utterances whose target is a
        # measurement rather than a summary of disagreement.
        self.aux_vad_agreement_weight = False
        # Permuted-label control: replace each sample's cluster ID with a
        # deterministic pseudo-random label derived by hashing its VAD values.
        # Spatially white, so the aux task keeps its shape and difficulty but
        # carries no usable VAD structure. If this arm matches the k-means
        # arm, the aux benefit is pure regularization.
        self.aux_vad_permute_clusters = False

        self.use_proto_atyp_split = False
        self.proto_atyp_split_criterion = "per_class_median"
        # "theoretical" / "class_means" -> VAD distance to a class centre
        # "nrc"        -> distance to the class mean NRC lexical profile
        # "random"     -> placebo, a hash of the transcript
        # "confidence" -> the model's own EMA confidence, frozen into a
        #                 split at confidence_freeze_epoch. The only
        #                 definition that is computable on an unseen corpus
        #                 without annotations, and the strongest signal in
        #                 the failure analysis (d = -1.2 to -1.5 separating
        #                 leaked-to-neutral from correctly classified).
        self.proto_atyp_center_source = "theoretical"
        # Epochs of plain 4-way training before the confidence split is
        # frozen. Early-epoch confidence reflects optimization state more
        # than sample difficulty, and a split that churns every epoch is a
        # moving target. Until this epoch the widened head is trained via
        # collapsed class probabilities, so only the parent task is active.
        self.confidence_freeze_epoch = 5
        self.confidence_momentum = 0.9
        # "margin"    -> top-1 minus top-2 probability (boundary proximity)
        # "true_prob" -> probability on the correct class (how wrong it is)
        self.confidence_metric = "margin"

        # Intensity-dependent soft targets toward neutral. The confusion
        # analysis found utterances leaking into neutral were acoustically
        # weaker (energy d = -0.41 angry, -0.29 happy; pitch d = -0.84
        # angry) and twice as likely to have a human annotator who also
        # said neutral (odds ratio 2.43). That suggests neutral is not a
        # peer category but the region where no emotion registers strongly,
        # so a weak happy utterance is genuinely part neutral and a one-hot
        # target asserts otherwise. Intensity shapes the training target
        # only; the model never estimates it at test time, which is why
        # using VAD here does not inherit the transfer failure that sank
        # the VAD-based prototypicality splits.
        self.use_neutral_soft_labels = False
        self.neutral_soft_alpha = 0.3
        # Soft CE target on the widened head that reduces the penalty for
        # confusing the two sub-labels of the same primary class.
        # target[correct_sub] = 1 - w, target[sibling_sub] = w, else 0.
        # w = 0.0 recovers hard CE. w = 0.5 removes any incentive to
        # distinguish proto from atyp within the correct primary class while
        # still fully punishing cross-class mistakes.
        self.proto_atyp_sibling_weight = 0.0

        # How hard the primary CE corrects for training-corpus class
        # imbalance. MSPP is 52% neutral / 8.5% sad, so this matters.
        #   "inverse_freq"      w ~ 1 / freq. Matches balanced risk (what UAR
        #                       measures) but scales the minority-class
        #                       gradient by ~6x, injecting variance.
        #   "sqrt_inverse_freq" w ~ 1 / sqrt(freq). Lower variance, but
        #                       under-corrects relative to UAR.
        #   "none"              w = 1. Use together with logit adjustment.
        self.class_weight_mode = "inverse_freq"
        # Logit adjustment (Menon et al., ICLR 2021). Adds tau * log(prior)
        # to the logits during training only; eval uses raw logits. This is
        # Fisher-consistent for balanced error and, unlike reweighting,
        # shifts the decision boundary rather than amplifying minority
        # gradients. Pair with class_weight_mode: none so imbalance is not
        # corrected twice.
        # Class-dependent margin (LDAM, Cao et al. 2019). Distinct from
        # inverse-frequency weighting: weighting corrects the prior and can
        # be satisfied by shifting the boundary, whereas a margin demands
        # the true class beat the others by a gap, which a shift cannot
        # produce. Composes with class_weight_mode rather than replacing it.
        self.use_ldam_margin = False
        self.ldam_max_margin = 0.5
        # LDAM assumes normalised logits at scale 30; this head is
        # unnormalised, so lower values may be needed for stability.
        self.ldam_scale = 30.0

        self.use_logit_adjustment = False
        self.logit_adjustment_tau = 1.0

        # Evidence heads: neutral as absence rather than a class. The head
        # becomes num_classes - 1 sigmoid outputs, one per emotional class.
        # Neutral samples train every head toward zero and never supply
        # positive evidence, so the model cannot learn a neutral prototype.
        # At eval the strongest head wins if it clears evidence_threshold,
        # else the prediction is neutral. Motivated by the SAMSEMO failure
        # analysis (19.5 percent of emotional utterances predicted neutral)
        # and by the failure of soft labels TOWARD neutral (-1.58 pts):
        # this pushes the opposite direction by construction.
        # Hierarchical head: column 0 is an emotion-presence logit, the
        # rest are emotion logits. P(neutral) = 1 - P(emotional) and
        # P(c) = P(emotional) * P(c | emotional), so argmax over the
        # composed distribution needs no threshold. Neutral samples train
        # the detector but give no gradient to the emotion head.
        # Latent sub-prototypes: widen the head to num_classes * K but
        # supervise only the parent class, collapsing prototype
        # probabilities before the loss. The split across prototypes is
        # never told to the model. Unlike every imposed prototypicality
        # definition tried here (VAD distance, lexical distance, frozen
        # confidence), there is no external label that has to transfer.
        self.use_latent_prototypes = False
        self.prototypes_per_class = 2
        # Optional bonus on the entropy of prototype usage. Off by
        # default: forcing samples to spread across prototypes imposes the
        # very structure this design exists to avoid assuming. Collapse
        # onto one prototype per class is a valid negative answer.
        self.prototype_usage_entropy = 0.0

        self.use_hierarchical_head = False
        self.hierarchical_detector_weight = 1.0
        # Optional positive weight on the presence detector. None leaves
        # the near-balanced binary problem alone; >1 makes the model more
        # willing to call something emotional, <1 more conservative.
        self.hierarchical_detector_pos_weight = None

        self.use_evidence_heads = False
        self.evidence_threshold = 0.5
        # Each one-vs-rest head sees 10-28 percent positives on MSPP, so
        # plain BCE biases every head toward "absent", the sigmoids sit low,
        # and samples fall through to neutral. pos_weight = n_neg / n_pos
        # corrects that; the cap stops the rarest class dominating.
        self.evidence_pos_weight_cap = 10.0

        # Anti-neutral margin (standard softmax head only): hinge penalty
        # on emotional samples whose true-class logit does not clear the
        # neutral logit by anti_neutral_margin. Only emotional samples
        # contribute.
        self.use_anti_neutral_margin = False
        self.anti_neutral_margin = 1.0
        self.anti_neutral_weight = 0.5

        # Ambiguity weighting: scale per-sample loss by the model's own
        # accumulated uncertainty (ConfidenceTracker EMA), w = 1 + beta *
        # (1 - confidence). Ambiguous samples get up to (1 + beta) weight.
        # ambiguity_shuffle is the placebo: a fixed permutation of the
        # weights that keeps their distribution but destroys their
        # per-sample meaning.
        self.use_ambiguity_weighting = False
        self.ambiguity_beta = 1.0
        self.ambiguity_warmup_epochs = 3
        self.ambiguity_shuffle = False

        # Early stopping
        self.early_stopping_patience = 10
        # Set False to train a fixed num_epochs budget with no early stop.
        # Validation UAR on the training corpus was measured to be
        # uncorrelated with cross-corpus test UAR (r = -0.05 over 15 runs),
        # so early stopping on it selects a checkpoint essentially at
        # random with respect to the metric of interest. A fixed budget
        # removes that selection noise from arm comparisons.
        self.use_early_stopping = True
        # Which weights to evaluate on the test corpora.
        #   "best_val" -> checkpoint with the highest val metric (default,
        #                 the historical behavior)
        #   "final"    -> weights at the end of the last epoch
        #   "swa_last_n" -> uniform average of the weights from the last
        #                 `swa_last_n` epochs. Averaging is over the raw
        #                 state dict, so it only makes sense on a fixed
        #                 budget where the last epochs are comparable.
        self.model_selection = "best_val"
        self.swa_last_n = 5

        # Logging
        self.wandb_project = "Emotion2Vec_Contrastive"
        self.experiment_name = "baseline"
        self.seed = 42
        self.seeds = [42]  # List of seeds for multi-seed runs

        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _fix_numeric_types(cls, config_dict):
        """Ensure numeric types are correctly parsed from YAML"""
        numeric_fields = [
            'learning_rate', 'weight_decay', 'dropout', 'batch_size', 'num_epochs',
            'audio_learning_rate', 'bert_learning_rate',
            'contrastive_weight', 'contrastive_temperature', 'prototypical_alpha',
            'prototypical_beta', 'prototypical_threshold', 'val_split',
            'audio_dim', 'hidden_dim', 'num_classes', 'text_max_length',
            'fusion_hidden_dim', 'num_attention_heads', 'vad_output_dim', 'seed',
            'projection_dim', 'projection_hidden_dim',
            'margin_base', 'separation_margin', 'separation_weight',
            'early_stopping_patience',
            'adversarial_weight', 'adversarial_alpha', 'adversarial_hidden_dim',
            'prototypical_weighting_alpha', 'salience_beta', 'salience_clip',
            'label_smoothing_beta', 'label_smoothing_max',
            'ce_weight_both_vad_w', 'learned_centroid_momentum', 'learned_centroid_lr',
            'proto_predictor_weight', 'proto_predictor_hidden_dim',
            'unfreeze_bert_layers', 'bert_learning_rate',
            'unfreeze_audio_layers', 'audio_learning_rate',
            'stage1_percentile', 'stage1_epochs', 'stage2_lr_factor',
            'max_audio_seconds', 'gradient_accumulation_steps',
            'mv_weight_vad', 'mv_weight_cross', 'mv_weight_embed',
            'hard_negative_weight',
            'bank_size', 'bank_momentum', 'bank_threshold',
            'cross_modal_dim', 'cross_modal_weight',
            'modality_adv_weight', 'adversarial_peak_lambda', 'adversarial_warmup_frac',
            'vadmix_alpha', 'vadmix_weight', 'vadmix_temperature',
            'vrex_lambda', 'vrex_warmup_frac',
            'curriculum_epochs', 'post_curriculum_dropout',
            'proto_atyp_sibling_weight',
            'aux_vad_cluster_k', 'aux_vad_cluster_weight', 'aux_vad_head_depth',
            'aux_vad_clusters_per_class', 'logit_adjustment_tau',
            'confidence_momentum', 'neutral_soft_alpha',
            'evidence_threshold', 'anti_neutral_margin', 'anti_neutral_weight',
            'ambiguity_beta', 'evidence_pos_weight_cap',
            'hierarchical_detector_weight', 'prototype_usage_entropy',
        ]

        for field in numeric_fields:
            if field in config_dict:
                try:
                    config_dict[field] = float(config_dict[field])
                except (ValueError, TypeError):
                    pass

        int_fields = [
            'batch_size', 'num_epochs', 'num_classes', 'text_max_length',
            'audio_dim', 'hidden_dim', 'fusion_hidden_dim', 'num_attention_heads',
            'vad_output_dim', 'seed', 'projection_dim', 'projection_hidden_dim',
            'early_stopping_patience', 'adversarial_hidden_dim',
            'proto_predictor_hidden_dim',
            'unfreeze_bert_layers',
            'unfreeze_audio_layers', 'stage1_epochs', 'gradient_accumulation_steps',
            'bank_size', 'cross_modal_dim',
            'curriculum_epochs',
            'aux_vad_cluster_k', 'aux_vad_head_depth',
            'aux_vad_clusters_per_class', 'swa_last_n',
            'confidence_freeze_epoch', 'ambiguity_warmup_epochs',
            'num_frames', 'num_attn_heads_pool', 'attn_pool_hidden',
            'prototypes_per_class',
        ]

        for field in int_fields:
            if field in config_dict:
                try:
                    config_dict[field] = int(float(config_dict[field]))
                except (ValueError, TypeError):
                    pass

        return config_dict

    @classmethod
    def from_yaml(cls, yaml_path, experiment_id=None):
        """
        Load config from YAML file.

        Supports two formats:
        1. Single config: all keys at top level
        2. Multi-experiment: template_config + experiments list
           Uses YAML anchors (&template) and merge keys (<<: *template)

        Args:
            yaml_path: Path to YAML file
            experiment_id: int (index) or str (name) to select experiment.
                          None = load as single config (or error if multi-experiment)
        """
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Check if multi-experiment config
        if "experiments" in yaml_config:
            if experiment_id is None:
                print("Available experiments:")
                for i, exp in enumerate(yaml_config["experiments"]):
                    print(f"   {i}: {exp.get('name', exp.get('id', f'experiment_{i}'))}")
                raise ValueError(
                    "Multi-experiment config detected. Use --experiment <id> or --all"
                )

            # Find the specified experiment
            if isinstance(experiment_id, int):
                if 0 <= experiment_id < len(yaml_config["experiments"]):
                    config_dict = yaml_config["experiments"][experiment_id]
                else:
                    raise ValueError(f"Experiment index {experiment_id} out of range")
            else:
                # Find by name or id
                config_dict = None
                for exp in yaml_config["experiments"]:
                    if exp.get("id") == experiment_id or exp.get("name") == experiment_id:
                        config_dict = exp
                        break
                if config_dict is None:
                    raise ValueError(f"Experiment '{experiment_id}' not found")

            exp_name = config_dict.get('name', config_dict.get('id', experiment_id))
            print(f"Running experiment: {exp_name}")
        else:
            config_dict = yaml_config

        # Remove YAML-only keys that aren't config fields
        config_dict.pop('template_config', None)

        # Map 'name' to 'experiment_name' (YAML uses 'name', Config uses 'experiment_name')
        if 'name' in config_dict and 'experiment_name' not in config_dict:
            config_dict['experiment_name'] = config_dict.pop('name')
        elif 'name' in config_dict:
            config_dict.pop('name')

        config_dict = cls._fix_numeric_types(config_dict)
        return cls(**config_dict)

    @classmethod
    def list_experiments(cls, yaml_path):
        """List all experiments in a multi-experiment YAML file"""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if "experiments" not in yaml_config:
            return None

        experiments = []
        for i, exp in enumerate(yaml_config["experiments"]):
            experiments.append({
                'index': i,
                'name': exp.get('name', exp.get('id', f'experiment_{i}')),
            })
        return experiments

    def to_dict(self):
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_yaml(self, yaml_path):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __repr__(self):
        return f"Config({self.experiment_name})"
