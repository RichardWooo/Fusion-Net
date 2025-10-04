# Fusion-Net

This repository provides reference implementations for a feature-driven generative
adversarial pipeline dubbed **Fusion-Net**. The main components include:

- **FusionGenerator** – combines image features and latent noise with a multi-head
  cross-attention fusion block before progressively upsampling to 256×256 RGB images.
- **PatchGANDiscriminator** – scores overlapping 30×30 patches for real/fake decisions.
- **Feature Similarity Loss** – encourages generator outputs to retain encoded
  characteristics of the input images.
- **Training Loop** – trains the generator and discriminator with hinge loss and the
  feature similarity regularizer.
- **FID Metric** – measures the Fréchet distance between real and generated image
  distributions using InceptionV3 features.
- **Grad-CAM Utility** – visualizes class-specific saliency maps from classifier models.

All modules are built with PyTorch and designed to be composable for research and
experimentation on feature-conditioned image synthesis.
