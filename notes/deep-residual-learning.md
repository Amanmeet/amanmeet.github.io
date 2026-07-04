---
title: Deep Residual Learning for Image Recognition
date: 2025-11-20
tags:
  - vision
  - architectures
updates:
  - date: 2025-11-25
    note: Added connection to transformer residual streams
---

# Deep Residual Learning for Image Recognition

**Authors:** He et al. (2015)
**Venue:** CVPR 2016

---

## Key Idea

The paper introduces **residual connections** (skip connections) that allow training of very deep networks (100+ layers). The core insight is that it is easier to learn a residual mapping `F(x) = H(x) - x` than the original mapping `H(x)` directly.

## The Degradation Problem

Before ResNets, simply stacking more layers led to *higher* training error -- not just overfitting, but optimization difficulty. This was surprising because a deeper network should be at least as expressive as a shallower one (it could learn identity mappings for the extra layers).

## Residual Blocks

A residual block computes:

```
y = F(x, {W_i}) + x
```

Where `F` is typically two or three convolutional layers. The skip connection adds the input `x` directly to the output, so the layers only need to learn the *residual* -- the difference from identity.

### Why This Helps

- If the optimal mapping is close to identity, pushing residuals toward zero is easier than learning identity through stacked nonlinear layers
- Gradients flow more easily through skip connections (mitigating vanishing gradients)
- Enables ensemble-like behavior -- ResNets can be viewed as an ensemble of many shallow networks

## Results

- Won 1st place on ImageNet 2015 (3.57% top-5 error)
- Successfully trained networks with **152 layers** (8x deeper than VGG)
- Also improved results on CIFAR-10, COCO detection, and COCO segmentation

## Personal Notes

The elegance of this paper is in its simplicity. The residual connection is almost trivially simple to implement -- just add the input to the output -- yet it fundamentally changed what was possible in deep learning.

What I find most interesting is the connection to later work: residual streams in transformers are essentially the same idea, and understanding information flow through residual connections has become central to mechanistic interpretability. The "residual stream" view of transformers owes a direct debt to this paper.

The degradation problem they identified is also a great example of how empirical observations (deeper networks performing worse) can lead to fundamental architectural innovations.
