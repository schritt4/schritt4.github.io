---
layout: post
title: Contrastive Representation Learning
date: 2024-02-19 00:00:00 +0900
tags: Representation-Learning
---
**Original Post**: [https://lilianweng.github.io/posts/2021-05-31-contrastive/](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

* this unordered seed list will be replaced by the toc
{:toc}

Contrastive representation learning의 목적은 비슷한 sample pair는 가깝게, 비슷하지 않은 sample pair는 멀게 mapping하는 embedding space를 학습하는데 있다. Contrastive learning은 supervised setting과 unsupervised setting 모두에서 사용가능한데, 특히 unsupervised setting에서는 self-supervised learning을 위한 하나의 강력한 접근법이 될 수 있다.

# Contrastive Training Objectives
초기 Contrastive learning의 loss function은 각각 하나의 positive와 negative만이 포함되어 있었지만, 요즘은 다수의 positive와 negative pair를 하나의 배치에 포함시켜 training objective를 구성한다.

## Contrastive Loss
**Contrastive loss**는 Contrastive learning의 training objective 중 가장 고전적인 것이다.  

Input sample의 list $$\\{ \mathbf{x}_i \\}$$ 와 각각에 대응되는 label $$y_i \in \\{1, ... L \\}$$ 가 주어져 있다고 해보자. 이때 우리는 같은 class 내의 sample 끼리는 비슷한 embedding vector를, 다른 class의 sample 끼리는 서로 다른 embedding vector를 가지도록 하는 함수 $$f_{\theta}(.):\mathcal{X} \rightarrow \mathbb{R}^{d}$$ 를 학습시키고자 할 것이다. 그러므로, contrastive loss는 한 쌍의 sample $$(x_i, x_j)$$ 를 input으로 받고 sample들이 같은 class에 속해있을 때는 embedding distance를 최소화(반대의 경우에는 최대화)한다. contrastive loss를 수식으로 표현하면 다음과 같다:

$$\mathcal{L}_{cont} (\mathbf{x}_i, \mathbf{x}_j, \theta ) = \mathbb{1}\left[y_i = y_j\right]||f_{\theta}(\mathbf{x}_i) - f_{\theta}(\mathbf{x}_j)||^2_2 + \mathbb{1} \left[y_i \neq y_j\right] max(0, \epsilon - ||f_{\theta}(\mathbf{x}_i) - f_{\theta}(\mathbf{x}_j)||_2)^2$$

$$\epsilon$$ 은 hyperparameter로 다른 class에 속해있는 sample들 사이의 최소 거리를 의미한다.

## Triplet Loss
**Triplet loss**는 원래 FaceNet에서 제안된 것으로, 다른 포즈 혹은 각도에 있는 같은 사람을 인식하기 위해 사용되었다.

<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/triplet-loss.png" width="50%">
</p>
<p align="center" style="color:gray;">
    Fig. 1. Illustration of triplet loss given one positive and one negative per anchor. <br>
    (Image source: <a href="https://arxiv.org/abs/1503.03832">Schroff et al. 2015</a>)
</p>
<br>

하나의 anchor input $$\mathbf{x}$$ 가 주어졌을 때, 우리는 하나의 positive sample $$\mathbf{x}^+$$ 과 하나의 negative sample $$\mathbf{x}^-$$ 를 선택한다. 
이때 $$\mathbf{x}^+$$ 는 $$\mathbf{x}$$ 와 같은 class에 속해있는 sample을, $$\mathbf{x}^-$$ 는 $$\mathbf{x}$$ 와 다른 class에 속해있는 sample을 의미한다. Triplet loss는 anchor $$\mathbf{x}$$ 와 positive $$\mathbf{x}^+$$ 의 거리를 최소화하고 anchor $$\mathbf{x}$$ 와 negative $$\mathbf{x}^-$$ 의 거리를 최대화하는 방식으로 학습한다. Triplet loss의 수식은 다음과 같다:

$$\mathcal{L}_{triplet}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \underset{x \in \mathcal{X}}{\sum} max(0, ||f(\mathbf{x}) - f(\mathbf{x}^+)||^2_2 - ||f(\mathbf{x}) - f(\mathbf{x}^-)||^2_2 + \epsilon)$$

margin parameter $$\epsilon$$ 은 비슷한 pair와 비슷하지 않은 pair 사이의 최소 거리 차이를 의미한다.

Triplet loss를 사용하는 경우, challenging(구분이 어려운) negative $$\mathbf{x}^-$$ 를 선택하는 것이 모델 성능 개선에 도움을 줄 수 있다.

## Lifeted Structured Loss
**Lifted Structured Loss**는 효율적인 연산을 위해 하나의 training batch에 존재하는 모든 pairwise edge를 사용한다.

<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/lifted-structured-loss.png" width="50%">
</p>
<p align="center" style="color:gray;">
    Fig. 2. Illustration compares contrastive loss, triplet loss and lifted structured loss.<br> Red and blue edges connect similar and dissimilar sample pairs respectively.<br>
    (Image source: <a href="https://arxiv.org/abs/1511.06452">Song et al. 2015</a>)
</p>
<br>

$$D_{ij}=||f(\mathbf{x}_i) - f(\mathbf{x}_j)||_2$$ 라고 하자.
이때 structured loss function은 다음과 같이 정의된다:

$$\begin{eqnarray} 
\mathcal{L}_{struct} = \frac {1} {2|\mathcal{P}|}\underset {(i, j) \in \mathcal{P}}{\sum} max(0, \mathcal{L}_{struct}^{(ij)})^2 \\
where\ \mathcal{L}^{(ij)}_{struct} = D_{ij} + max(\underset{(i, k) \in \mathcal{N}}{max} \epsilon - D_{ik}, \underset {(j,l) \in \mathcal{N}}{max}\epsilon - D_{jl})
\end{eqnarray}$$

여기에서 $$\mathcal {P}$$ 는 positive sample을 포함하는 집합을, $$\mathcal {N}$$ 은 negative sample을 포함하는 집합을 의미한다. 위의 수식에 포함되어 있는 dense pairwise squared distance matrix는 training batch 별로 쉽게 계산할 수 있다.

$$\mathcal{L}_{struct}^{(ij)}$$ 에서 $$D_{ij}$$ 뒤에 있는 부분은 challenging한 negative sample을 찾기 위해 사용된다. 그러나, max 함수는 연속함수가 아니기 때문에 나쁜 local optimum으로의 수렴을 야기할 수 있다. 따라서, 우리는 위의 식을 다음과 같이 relax할 수 있다:

$$\mathcal{L}_{struct}^{(ij)} = D_{ij} + \log{\left(\sum_{(i,k) \in \mathcal{N}} \exp(\epsilon - D_{ik}) + \sum_{(j, l) \in \mathcal {N}} \exp (\epsilon - D_{jl})\right)}$$

## N-pair Loss
**Multi-Class N-pair loss**는 다수의 negative samples과의 비교를 포함한, triplet loss의 일반화된 형태라고 볼 수 있다.

$$(N+1)$$개의 tuplet 학습 데이터, $$\left\{ \mathbf{x}, \mathbf{x}^{+}, \mathbf{x}^{-}_{1}, ..., \mathbf{x}^{-}_{N-1} \right\}$$, 가 있다고 하자. 이때 학습 데이터에는 한 개의 positive sample과 $$N-1$$ 개의 negative sample이 포함된다. N-pair loss는 다음과 같이 정의된다:

$$\begin{eqnarray}
\mathcal{L}_{N-pair}(\mathbf{x}, \mathbf{x}^+, \{\mathbf{x}^-_i\}^{N-1}_{i=1}) & = & \log{\left(1+\sum^{N-1}_{i=1} \exp (f(\mathbf{x})^\top f(\mathbf{x}^-_i) - f(\mathbf{x})^\top f(\mathbf{x}^+)) \right)} \\
& = & - \log \frac {\exp (f(\mathbf{x})^\top f(\mathbf{x}^+))} {\exp(f(\mathbf{x})^\top f(\mathbf {x}^+)) + \sum^{N-1}_{i=1} \exp{(f(\mathbf{x})^\top f(\mathbf{x}^-_i))}}
\end{eqnarray}$$

만약 class 별로 한 개의 negative sample을 추출한다면, multi-class classification의 softmax loss와 같아진다.

## NCE
**Noise Contrastive Estimation**은 통계 모델의 parameter를 추정하기 위해 사용된다. NCE의 아이디어는 로지스틱 회귀분석을 사용하여 target data와 noise를 구분하는 것이다.

$$\mathbf{x} \sim P(\mathbf{x}\\|C=1;\theta) = p_\theta(\mathbf{x})$$, $$\tilde {\mathbf{x}} \sim P(\tilde {\mathbf{x}}\\|C=0)=q(\tilde {\mathbf{x}})$$ 라고 하자. 이때 $$\mathbf {x}$$ 는 target sample, $$\tilde {\mathbf {x}}$$ 는 noise sample을 의미한다. 우리는 noise distribution 대신에 target data distribution에서 sampling한 sample $$u$$의 logit을 모델링 할 것이다:

$$\mathcal{l}_\theta(\mathbf{u}) = \log \frac {p_\theta (\mathbf{u})}{q(\mathbf{u})} = \log p_\theta(\mathbf{u}) - \log q(\mathbf{u})$$

sigmoid 함수 $$\sigma(\cdot)$$ 를 사용하여 logit을 확률로 변환한 후에, 우리는 cross entropy loss를 적용할 수 있다:

$$\begin{eqnarray} 
\mathcal{L}_{NCE} = -\frac {1}{N} \sum_{i=1}^{N} \left [ \log \sigma(\mathcal{l}_\theta (\mathbf{x}_i)) + \log(1 - \sigma(\mathcal{l}_\theta(\tilde{\mathbf{x}}_i))) \right]\\ 
where \ \sigma(\mathcal{l}) = \frac{1}{1+\exp(-\mathcal{l})} = \frac{p_\theta}{p_\theta+q} 
\end{eqnarray}$$

위의 형태는 하나의 positive sample과 하나의 noise sample만을 사용하는 원래의 NCE loss이다. 하지만 많은 후속 연구에서, multiple negative sample을 포함하는 contrastive loss의 경우에도 NCE라는 용어를 사용한다.

## InfoNCE
**InfoNCE loss**는 NCE에서 영감을 받은 것으로, Contrastive Predictive Coding이라는 unsupervised learning 기법에서 사용되는 loss이다. InfoNCE는 cateogorical cross-entropy loss를 사용하여 noise sample들 중 positive sample을 식별할 수 있도록 구성되었다.

Context vector $$\mathbf{c}$$가 주어졌다고 해보자. 이때, positive sample의 분포는 $$p(\mathbf{x}\vert\mathbf{c})$$, negative sample의 경우에는 context vector $$c$$와 독립적인 $$p(\mathbf{x})$$로 간주한다. 간단하게, 모든 sample을 $$X=\{\mathbf{x}_i\}^N_{i=1}$$라고 표현할 수 있고, 이중에는 한 개의 positive sample $$\mathbf{x}_{pos}$$이 있다고 해보자. 우리가 positive sample을 잘 찾을 확률은 다음과 같이 표현할 수 있다:

$$p(C=pos\vert X, \mathbf{c}) = \frac {p(x_{pos} \vert \mathbf{c})\prod_{i=1,...N ;i\neq pos}p(\mathbf{x}_i)}{\sum^N_{j=1}\left[p(\mathbf{x}_j\vert \mathbf{c} \prod_{i=1, ..., N;i\neq j}p(\mathbf{x}_i))\right]}=\frac {\frac{p(\mathbf{x}_{pos}\vert \mathbf{c})}{p(\mathbf{x}_{pos})}}{\sum^N_{j=1} \frac{p(\mathbf {x}_j \vert \mathbf{c})}{p(\mathbf{x}_j)}}=\frac {f(\mathbf{x}_{pos}, \mathbf{c})}{\sum^N_{j=1}f(\mathbf {x}_j, \mathbf{c})}$$

이때 scoring function은 $$f(\mathbf{x}, \mathbf{c}) \propto \frac{p(\mathbf{x}\vert \mathbf{c})}{p(\mathbf{x})} $$를 따른다.

InfoNCE loss의 경우 위에서 구한 확률에 negative log를 씌운 형태를 최적화하게 된다:

$$\mathcal{L}_{InfoNCE}=-\mathbb{E}\left[{\log{\frac{f(\mathbf{x}, \mathbf{c})}{\sum_{\mathbf{x}'\in X}f(\mathbf{x}', \mathbf{c})}}}\right]$$

$$f(\mathbf{x}, \mathbf{c})$$가 density ratio $$\frac{p(\mathbf{x}\vert \mathbf{c})}{p(\mathbf{x})}$$를 추정한다는 사실은 mutual information optimization과 연관이 있다. input $$\mathbf{x}$$와 context vector $$\mathbf{c}$$의 mutual information을 최대화하는 것이 위에서 구한 InfoNCE loss를 최적화 하는 것과 동치라는 것을 쉽게 알 수 있다:

$$I(\mathbf{x}; \mathbf{c})=\sum_{\mathbf{x}, \mathbf{c}}p(\mathbf{x},\mathbf{c})\log{\frac{p(\mathbf{x},\mathbf{c})}{p(\mathbf{x})p(\mathbf{c})}}=\sum_{\mathbf{x}, \mathbf{c}}p(\mathbf{x},\mathbf{c})\log{\frac{p(\mathbf{x}\vert \mathbf{c})}{p(\mathbf{x})}}$$

이때 마지막 logarithmic term은 앞서 구한 f와 연관이 있음을 기억하자.

CPC는 Sequence prediction task에서 직접적으로 future observation의 확률 $$p_k(\mathbf{x}_{t+k}\vert \mathbf{c}_t)$$를 모델링하는 것이 아니라, $$\mathbf{x}_{t+k}$$와 $$\mathbf{c}_t$$ 사이의 mutual information를 보존하는 density ratio를 모델링한다:

$$f_k(\mathbf{x}_{t+k}, \mathbf{c}_t)=\exp{(\mathbf{z}_{t+k}^\top\mathbf{W}_k\mathbf{c}_t})\propto \frac{p(\mathbf{x}_{t+k}\vert \mathbf{c}_t)}{p(\mathbf{x}_{t+k})}$$

이때 $$\mathbf{z}_{t+k}$$는 encoding된 input, $$\mathbf{W}_k$$는 학습 가능한 weight matrix이다.

## Soft-Nearest Neighbors Loss
**Soft-Nearest Neighbors Loss**는 positive sample이 하나 이상 있는 경우로 확장된 경우이다.

batch sample $$\{(\mathbf{x}_i, y_i)\}_{i=1}^B$$가 주어져 있다고 하자. 이때 $$y_i$$는 $$\mathbf{x}_i$$의 class label이고, 함수 $$f(\cdot, \cdot)$$은 두 개의 input의 거리를 측정할 때 사용된다. 이러한 setting에서 temperature $$\tau$$에서의 soft nearest neighbor loss를 정의하면 다음과 같다:

$$\mathcal{L}_{snn}=-\frac{1}{B}\sum^B_{i=1} \log {\frac {\sum_{i\neq j, y_i=y_j,j=1,..,B} \exp{(-f(\mathbf{x}_i, \mathbf{x}_j)/\tau)}}{\sum_{i\neq k, k=1,..,B} \exp{(-f(\mathbf{x}_i, \mathbf{x}_k)/\tau)}}}$$

temperature $$\tau$$는 representation space에서 같은 class의 feature vector들을 얼마나 밀집하게 위치시킬 것인가를 결정하는 계수이다. 예를 들어, low temperature의 경우에는 loss값이 가까운 거리의 input들에게 더 큰 영향을 받게 되고, 거리가 먼 경우에는 loss에 큰 영향을 주지 못하게 된다.

## Common Setup
우리는 unsupervised data에서 positive sample과 negative sample을 생성하기 위해 soft nearest neighbor loss에서의 "classes"와 "labels"를 재정의할 필요가 있다. 예를 들면 data augmentation을 통해 original sample의 noise 버전을 생성하는 것처럼 말이다.

요즘의 연구들은 다수의 positive sample과 negative sample을 한 번에 고려하기 위해 앞으로 언급할 contrastive learning objective의 정의를 따른다. $$p_{data}(\cdot)$$는 $$\mathbb{R}^n$$에서의 data 분포를, $$p_{pos}(\cdot, \cdot)$$는 $$\mathbb{R}^n$$에서의 positive pair의 분포를 의미한다고 하자. 이 두 분포들은 다음을 만족한다:

- Symmetry: $$\forall \mathbf{x}, \mathbf{x}^+, p_{pos}(\mathbf{x}, \mathbf{x}^+)=p_{pos}(\mathbf{x}^+, \mathbf{x})$$
- Matching marginal: $$\forall \mathbf{x}, \int p_{pos} (\mathbf{x}, \mathbf{x}^+)d\mathbf{x}^+ = p_{data}(\mathbf{x})$$

인코더 $$f(\mathbf{x})$$로 하여금 L-2 normalized feature vector를 학습하게 하기 위해서는 다음과 같은 contrastive learning objective를 설정해주면 된다:

$$
\begin{eqnarray} 
\mathcal{L_{contrastive} = \mathbb{E}_{(\mathbf{x}, \mathbf{x}^+)\sim p_{pos}, \{\mathbf{x}_i^-\}_{i=1}^M \overset{\text{i.i.d}}{\sim}p_{data}}}\left[- \log{\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)/\tau)}{\exp{(f(\mathbf{x})^\top f(\mathbf{x}^+)/\tau)}+\sum^M_{i=1}\exp{(f(\mathbf{x})^\top f(\mathbf{x}_i^-)/\tau)}}} \right] \\ 
\simeq \mathbb{E}_{(\mathbf{x}, \mathbf{x}^+)\sim p_{pos}, \{\mathbf{x}_i^-\}_{i=1}^M \overset{\text{i.i.d}}{\sim}p_{data}}\left[-\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)/\tau)+\log{(\sum^M_{i=1}\exp{(f(\mathbf{x})^\top f(\mathbf{x}_i^-)/\tau)})} \right] \\
= - \frac{1}{\tau} \mathbb{E}_{(\mathbf{x}, \mathbf{x}^+)~p_{pos}} \left [f(\mathbf{x})^\top f(\mathbf{x}^+) \right ] + \mathbb{E}_{\mathbf{x}\sim p_{data}} \left[\log \mathbb{E}_{\mathbf{x}^- \sim p_{data}} \left[\sum^M_{i=1} \exp {(f(\mathbf{x})^\top f(\mathbf{x}_i^-)/\tau)}\right] \right]
\end{eqnarray}$$