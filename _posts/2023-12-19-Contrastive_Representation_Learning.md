---
layout: post
title: Contrastive Representation Learning
date: 2023-12-19 01:26:00 +0900
category: 
- Machine_Learning
- Representation_Learning
use_math: true
---
**Original Post**: [https://lilianweng.github.io/posts/2021-05-31-contrastive/](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
<br><br>
Contrastive representation learning의 목적은 비슷한 sample pair는 가깝게, 비슷하지 않은 sample pair는 멀게 mapping하는 embedding space를 학습하는데 있다. Contrastive learning은 supervised setting과 unsupervised setting 모두에서 사용가능한데, 특히 unsupervised setting에서는 self-supervised learning을 위한 하나의 강력한 접근법이 될 수 있다.

# Contrastive Training Objectives
초기 Contrastive learning의 loss function은 각각 하나의 positive와 negative만이 포함되어 있었지만, 요즘은 다수의 positive와 negative pair를 하나의 배치에 포함시켜 training objective를 구성한다.

## Contrastive Loss
**Contrastive loss**는 Contrastive learning의 training objective 중 가장 고전적인 것이다.  
<br>
Input sample의 list $\\{ \mathbf{x}_i \\}$ 와 각각에 대응되는 label $y_i \in \\{1, ... L \\}$ 가 주어져 있다고 해보자.

이때 우리는 같은 class 내의 sample 끼리는 비슷한 embedding vector를, 다른 class의 sample 끼리는 서로 다른 embedding vector를 가지도록 하는 함수 $f_{\theta}(.):\mathcal{X} \rightarrow \mathbb{R}^{d}$ 를 학습시키고자 할 것이다. 그러므로, contrastive loss는 한 쌍의 sample $(x_i, x_j)$ 를 input으로 받고 sample들이 같은 class에 속해있을 때는 embedding distance를 최소화(반대의 경우에는 최대화)한다. contrastive loss를 수식으로 표현하면 다음과 같다:

$$\mathcal{L}_{cont} (\mathbf{x}_i, \mathbf{x}_j, \theta ) = \mathbb{1}\left[y_i = y_j\right]||f_{\theta}(\mathbf{x}_i) - f_{\theta}(\mathbf{x}_j)||^2_2 + \mathbb{1} \left[y_i \neq y_j\right] max(0, \epsilon - ||f_{\theta}(\mathbf{x}_i) - f_{\theta}(\mathbf{x}_j)||_2)^2$$

$\epsilon$ 은 hyperparameter로 다른 class에 속해있는 sample들 사이의 최소 거리를 의미한다.

## Triplet Loss
**Triplet loss**는 원래 FaceNet에서 제안된 것으로, 다른 포즈 혹은 각도에 있는 같은 사람을 인식하기 위해 사용되었다.

<p align="center">
  <img src="{{site.baseurl}}/assets/images/triplet-loss.png" width="50%">
</p>
<p align="center" style="color:gray;">
    Fig. 1. Illustration of triplet loss given one positive and one negative per anchor. <br>
    (Image source: <a href="https://arxiv.org/abs/1503.03832">Schroff et al. 2015</a>)
</p>
<br>

하나의 anchor input $\mathbf{x}$ 가 주어졌을 때, 우리는 하나의 positive sample $\mathbf{x}^+$ 과 하나의 negative sample $\mathbf{x}^-$ 를 선택한다. 
이때 $\mathbf{x}^+$ 는 $\mathbf{x}$ 와 같은 class에 속해있는 sample을, $\mathbf{x}^-$ 는 $\mathbf{x}$ 와 다른 class에 속해있는 sample을 의미한다. Triplet loss는 anchor $\mathbf{x}$ 와 positive $\mathbf{x}^+$ 의 거리를 최소화하고 anchor $\mathbf{x}$ 와 negative $\mathbf{x}^-$ 의 거리를 최대화하는 방식으로 학습한다. Triplet loss의 수식은 다음과 같다:

$$\mathcal{L}_{triplet}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \underset{x \in \mathcal{X}}{\sum} max(0, ||f(\mathbf{x}) - f(\mathbf{x}^+)||^2_2 - ||f(\mathbf{x}) - f(\mathbf{x}^-)||^2_2 + \epsilon)$$

margin parameter $\epsilon$ 은 비슷한 pair와 비슷하지 않은 pair 사이의 최소 거리 차이를 의미한다.
<br><br>
Triplet loss를 사용하는 경우, challenging(구분이 어려운) negative $\mathbf{x}^-$ 를 선택하는 것이 모델 성능 개선에 도움을 줄 수 있다.

## Lifeted Structured Loss
**Lifted Structured Loss**는 효율적인 연산을 위해 하나의 training batch에 존재하는 모든 pairwise edge를 사용한다.

<p align="center">
  <img src="{{site.baseurl}}/assets/images/lifted-structured-loss.png" width="50%">
</p>
<p align="center" style="color:gray;">
    Fig. 2. Illustration compares contrastive loss, triplet loss and lifted structured loss.<br> Red and blue edges connect similar and dissimilar sample pairs respectively.<br>
    (Image source: <a href="https://arxiv.org/abs/1511.06452">Song et al. 2015</a>)
</p>
<br>
$D_{ij}=||f(\mathbf{x}_i) - f(\mathbf{x}_j)||_2$ 라고 하자.
이때 structured loss function은 다음과 같이 정의된다:

$$\mathcal{L}_{struct} = \frac {1} {2|\mathcal{P}|}\underset {(i, j) \in \mathcal{P}}{\sum} max(0, \mathcal{L}_{struct}^{(ij)})^2$$

$$ where\ \mathcal{L}^{(ij)}_{struct} = D_{ij} + max(\underset{(i, k) \in \mathcal{N}}{max} \epsilon - D_{ik}, \underset {(j,l) \in \mathcal{N}}{max}\epsilon - D_{jl})$$