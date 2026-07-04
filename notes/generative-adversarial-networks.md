---
title: Generative Adversarial Networks
date: 2025-10-08
tags:
  - generative models
  - GANs
updates: []
---

# Generative Adversarial Networks

**Authors:** Goodfellow et al. (2014)
**Venue:** NeurIPS 2014

---

## Key Idea

GANs introduce a **minimax game** between two networks: a **generator** `G` that produces synthetic data and a **discriminator** `D` that tries to distinguish real data from generated data. Through this adversarial process, `G` learns to produce increasingly realistic samples.

## Framework

The objective function:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

Where:
- `x` is sampled from the real data distribution
- `z` is sampled from a prior noise distribution (typically Gaussian)
- `G(z)` maps noise to data space
- `D(x)` outputs the probability that `x` is real

### Training Procedure

1. Update discriminator: maximize ability to distinguish real from fake
2. Update generator: minimize discriminator's ability to detect fakes
3. Alternate between these steps

At equilibrium, `G` produces samples from the true data distribution, and `D` outputs 0.5 everywhere (cannot tell real from fake).

## Strengths

- No explicit density estimation needed -- learns to sample directly
- Produces sharp, high-quality samples (compared to VAEs at the time)
- Very general framework -- applicable to images, text, audio, etc.

## Weaknesses / Challenges

- **Mode collapse**: Generator may learn to produce only a few types of samples
- **Training instability**: The minimax game can oscillate rather than converge
- **No evaluation metric**: Hard to quantify generation quality (FID/IS came later)
- **No latent inference**: Unlike VAEs, no built-in way to encode data back to latent space

## Subsequent Work

The GAN framework spawned an enormous body of follow-up work:
- DCGAN (2015): Convolutional architecture guidelines
- WGAN (2017): Wasserstein distance for stable training
- StyleGAN (2019): High-fidelity face generation
- BigGAN (2019): Large-scale class-conditional generation

## Personal Notes

This paper is a masterclass in framing. The game-theoretic perspective on generative modeling was genuinely novel and opened up an entirely new way of thinking about the problem. The minimax formulation is clean and mathematically elegant.

However, I think the practical challenges (mode collapse, training instability) turned out to be more fundamental than the original paper suggested. It took years of engineering tricks and architectural innovations before GANs became reliably trainable.

It's also interesting to consider GANs in the context of the current moment: diffusion models have largely superseded GANs for image generation, suggesting that the adversarial training paradigm, while beautiful, may not be the most practical path to high-quality generation.
