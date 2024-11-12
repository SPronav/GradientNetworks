# GradientNetworks

Code for Gradient Networks (https://arxiv.org/abs/2404.07361).

Gradient networks (GradNets) are neural networks guaranteed to correspond to the gradient of a potential function, i.e., they are conservative vector fields. Monotone gradient networks (mGradNets) correspond to gradients of convex functions. 

We provide code for cascaded (monotone) gradient networks (GradNet-C, mGradNet-C) that can universally approximate the gradient of any function expressible as the sum of (convex) ridge functions. We also provide code for modular (monotone) gradient networks (GradNet-M, mGradNet-M) that can universally approximate the gradient of any smooth (convex) function. For more details on the approximation capabilities of these networks, please see the paper.
