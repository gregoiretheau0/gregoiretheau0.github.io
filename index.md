---
layout: default
title: Home
---

<div class="profile-header" style="margin-bottom: 2rem;">
  <h1 style="margin-bottom: 0.2rem;">Grégoire Théau</h1>
  <p style="font-size: 1.15rem; color: #555; margin-top: 0;">
    <strong>Apprentice at Airbus AI Research</strong> | M.Sc. from <strong>CentraleSupélec</strong>
  </p>
  <p style="font-size: 1.05rem; line-height: 1.6;">
    Investigating <strong>Trustworthy AI</strong>, <strong>Geometric Robustness</strong>, and <strong>Certifiable Vision-Based Perception</strong> for safety-critical aerospace systems.
  </p>
</div>

## About Me

I am a Apprentice at **Airbus AI Research**, working at the intersection of deep learning and safety-critical avionics. I graduated with a Master of Science degree from **CentraleSupélec**.

My research focuses on evaluating the operational safety of multi-stage perception chains—specifically bridging keypoint regression networks (e.g., YOLOv8-Pose) with non-linear geometric solvers (Perspective-n-Point) under physically plausible operational disturbances such as camera rotations and lighting shifts.


## Research Interests

* **Spatial & Geometric Robustness**: Evaluating perception pipelines against non-convex physical perturbations beyond standard \(L_p\)-norm noise.
* **Vision-Based Navigation & Landing (VBL)**: Aircraft pose estimation and runway keypoint regression.
* **Global Optimization for AI Evaluation**: Applying Global Lipschitzian Optimization (GLO) to isolate critical failure modes and efficiently prune validation search spaces.
* **Differentiable Pose Estimation**: End-to-end differentiable PnP solvers (BPnP) and common PnPs solvers comparing.

---

{% comment %}
## Publications & Submissions

<div style="background-color: #f8f9fa; border-left: 4px solid #0052cc; padding: 1.2rem; border-radius: 4px; margin-bottom: 2rem;">
  <h3 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.15rem;">
    Robust Validation to Geometric Perturbations for Autonomous Pose Estimation
  </h3>
  <p style="font-size: 0.95rem; color: #666; margin-bottom: 0.8rem;">
    <em>Submitted to ECCV 2026 Workshop</em> <!-- [Paper PDF](#) | [Code](#) -->
  </p>
  <p style="font-size: 0.95rem; line-height: 1.5; margin-bottom: 0;">
    <strong>Abstract:</strong> Deploying vision models in safety-critical domains demands guaranteed robustness against physical geometric perturbations. We demonstrate that standard gradient-based attacks (e.g., APGD) fail on multi-stage pose estimation pipelines. By reformulating pose estimation robustness under <strong>Global Lipschitzian Optimization (GLO)</strong>, we successfully isolate critical failure modes while rapidly pruning over 80% of the worst-case search space in 10 iterations.
  </p>
</div>

{% endcomment %}

## Recent Posts

<ul class="post-list" style="list-style: none; padding-left: 0;">
  {% for post in site.posts %}
    <li style="margin-bottom: 1.5rem;">
      <span class="post-meta" style="color: #828282; font-size: 0.9rem;">{{ post.date | date: "%b %d, %Y" }}</span>
      <h3 style="margin-top: 0.2rem; margin-bottom: 0.4rem;">
        <a class="post-link" href="{{ post.url | relative_url }}" style="text-decoration: none; color: #0366d6;">
          {{ post.title | escape }}
        </a>
      </h3>
      {% if post.excerpt %}
        <p style="font-size: 0.95rem; color: #444; margin: 0;">
          {{ post.excerpt | strip_html | truncatewords: 30 }}
        </p>
      {% endif %}
    </li>
  {% endfor %}
</ul>