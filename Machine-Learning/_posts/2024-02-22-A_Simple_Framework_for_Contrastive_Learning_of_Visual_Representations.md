---
layout: post
title: A Simple Framework for Contrastive Learning of Visual Representations 리뷰
date: 2024-02-22 00:00:00 +0900
tags: Representation-Learning
---
* this unordered seed list will be replaced by the toc
{:toc}

# 1. Abstract & Contribution

본 논문은 image의 label이 주어지지 않은 상황에서 visual representation을 효과적으로 추출할 수 있는 SimCLR라는 알고리즘을 제안합니다. SimCLR의 특징적인 부분은 MoCo와 다르게 특별한 architecture나 memory bank가 필요하지 않다는 점입니다.

저자들은 이 연구에서 크게 3가지를 보이고자 하였습니다.

1. Data augmentation의 조합이 효과적인 predictive task 정의에 중요하다.
2. representation을 nonlinear layer에 통과시킨 후 contrastive loss를 계산하는 것이 더 좋은 quality의 representation을 학습하는데 도움이 된다.
3. contrastive learning은 supervised learning과 비교하였을 때, 더 큰 batch size와 더 많은 training step에서 더 좋은 성능을 보인다.

# 2. Method

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR.png" width="50%">
</p>
<br>

SimCLR의 알고리즘을 도식화하면 다음과 같습니다. $$x$$에 서로 다른 두 data augmentation $$(t \sim \mathcal T, t' \sim \mathcal T)$$을 적용하여, 같은 이미지에서 나온 두 개의 결과 $$(\tilde x_i, \tilde x_j)$$를 positive pair로 정의하고, 서로 다른 이미지에서 나온 결과들은 negative pair로 적용하여 contrastive learning 방식을 적용하고자 하였습니다. 
data augmentation을 거친 이미지 $$(\tilde x_i, \tilde x_j)$$는 base encoder $$f$$를 통과하여 visual representation vector로 변환이 됩니다. 이후, representation은 projection head $$g$$를 통과하여 contrastive loss가 적용되는 space로 mapping됩니다. 이때 논문에서는 base encoder로는 ResNet을, projection head로는 두 개의 linear layer 사이에 ReLU activation function을 넣은 구조의 MLP 네트워크를 사용하였습니다.


만약 batch size $$N$$의 minibatch를 training에 사용한다고 가정하면 data augmentation을 거쳤을 때 총 $$2N$$개의 sample을 얻게 됩니다. 즉, 한 개의 positive pair에 대해 $$2(N-1)$$개의 negative pair를 구성할 수 있게 됩니다. 논문에서는 NT-Xent loss를 사용하여 positive pair 간의 similarity는 최대화하고, negative pair 간의 similarity는 최소화하고자 하였습니다. NT-Xent loss는 다음과 같은 수식을 가집니다:


$$\begin{eqnarray}
\mathcal l_{i,j} = - \log{\frac {\exp (\text{sim}(\mathbf z_i,\mathbf z_j)/\tau)}{\sum^{2N}_{k=1}\mathbb 1_{[k\neq i]}\exp(\text{sim}(\mathbf z_i,\mathbf z_j)/\tau)}} \\
where ~~
\text {sim}(\mathbf u, \mathbf v)=\mathbf u^\top \mathbf v / \|\mathbf u\| \|\mathbf v\|
\end{eqnarray}
$$


Contrastive learning을 진행하기 위해서는 일반적으로 좋은 quality를 가지며 충분히 많은 양의 negative pair가 필요합니다. 따라서 본 논문에서는 큰 batch size를 이용하였고, 큰 batch size에서 불안정한 SGD나 Momentum optimizer 대신 LARS optimizer를 이용하였습니다. 또한 Global Batch Normalization을 사용함으로써, 분산 학습에서 단순히 Batch Normalization을 사용하였을 때 생길 수 있는 정보 손실을 방지하고자 하였습니다.

# 3. Data Augmentation for Contrastive Representation Learning

SimCLR 이전의 연구에서는 positive pair와 negative pair를 생성하기 위해 서로 다른 모델 구조를 가진 네트워크를 사용하였습니다. 예를 들면, receptive field를 달리하여 하나의 CNN 네트워크에서는 local한 정보를 추출하고, 다른 하나의 CNN 네트워크에서는 global한 정보를 추출하여 contrastive loss를 적용하는 식으로 학습을 진행했다고 합니다. 하지만 이는 단순하게 random crop을 진행하는 것과 같은 효과를 얻게 됩니다. 따라서 SimCLR에서는 supervised learning에서 일반적으로 사용하던 data augmentation 방법을 통해 훨씬 간단하게 positive pair와 negative pair를 생성합니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-augmentation.png" width="70%">
</p>
<br>

앞서 언급한 바와 같이 본 논문에서는 data augmentation의 조합이 좋은 representation을 생성하는데 중요한 역할을 한다는 것을 보이고자 하였습니다. 위의 그림은 논문에서 사용했던 data augmentation 방법들을 나열한 것입니다. SimCLR에서는 crop이나 resize, rotate, cutout 등 이미지의 공간적/기하학적 구조를 변형하는 방법과 color drop, jitter, Gaussian blur, Sobel filter 등 이미지의 색상을 왜곡하는 방법을 사용하였습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-augmentation2.png" width="70%">
</p>
<br>

논문에서는 총 7가지의 data augmentation 방법을 하나 또는 두 개 이어붙이는 식으로 실험을 진행하였습니다. 결과적으로는 하나의 augmentation 만으로는 좋은 성능을 달성하기 어려웠고, 여러 augmentation을 사용하여 predictive task를 높임으로써 더 좋은 quality의 representation을 얻을 수 있었다고 합니다. 표를 통해 알 수 있듯이 random crop 이후 color distortion을 진행하였을 경우에 가장 좋은 성능을 보여줬습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-color.png" width="70%">
</p>
<br>

특히 논문에서는 color distortion의 필요성을 역설하였습니다. 위의 그림은 서로 다른 두 이미지(첫 번째 행과 두 번째 행)에 대해서 random crop만을 진행하였을 때와 color distortion까지 적용하였을 때의 patch 들의 color distribution을 보여주는 histogram입니다. 이를 통해 color distortion을 적용하지 않는 경우에는 random crop된 patch들이 서로 같은 color distribution을 공유하고, 결국 네트워크는 이미지의 시각적인 특징에 주목하는 것이 아니라 color distribution에만 주목하여 generalizable feature를 학습하지 못하게 됩니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-color2.png" width="70%">
</p>
<br>

위의 표는 data augmentation의 세기에 따라 SimCLR의 성능이 어떻게 변하는지를 보여주고 있습니다. 이를 통해 color distortion의 강도가 세질수록 predictive task의 난이도가 증가하여 더 좋은 representation을 얻을 수 있고, 심지어 supervised learning에 도움이 되지 않는 강도의 augmentation도 SimCLR에서는 성능 향상에 기여할 수 있다는 것을 확인할 수 있습니다.

# 4. Architectures for Encoder and Head

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-architecture.png" width="50%">
</p>
<br>

위의 그림을 통해 모델의 크기가 커질 수록 SimCLR의 성능이 향상되고 있음을 확인할 수 있습니다. 특히, supervised counterpart와의 간극이 parameter 수가 증가함에 따라 작아지는 것을 확인할 수 있는데, 이를통해 우리는 supervised learning 보다 unsupervised learning에서 큰 모델을 사용하는 것이 더 큰 이점으로 작용한다는 것을 알 수 있습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-projection.png" width="50%">
</p>
<br>

또한 본 논문에서는 MoCo와 다르게 non-linear projection head를 도입하였습니다. 저자들은 linear projection head나 projection head를 아예 사용하지 않을 때보다 non-linear projection head를 통과시킨 후 contrastive loss를 계산하는 것이 더 좋은 성능을 낼 수 있음을 실험을 통해 증명하였습니다. 이때 projection head의 output dimension은 성능에 큰 영향을 주지 않는 것으로 밝혀졌습니다.

우리는 projection head를 통해 학습 성능을 개선시킬 수 있지만, 이를 통해 얻은 output vector는 시각적인 특징을 잘 반영하고 있다 보기 어렵습니다. 그 이유는 projection head가 data transformation과 무관하게 학습이 진행되기 때문인데, 결국 representation이 projection head를 통과하면서 downstream task 수행에 유용한 정보들을 잃게됩니다. 따라서 우리는 projection head를 학습에만 사용하고, 학습이 끝난 이후에는 encoder $f$만 사용하게 되는 것입니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-projection2.png" width="50%">
</p>
<br>

위의 표는 어떤 transformation이 적용되었는지 예측하는 task에서 $\mathbf h$와 $g(\mathbf h)$의 성능을 보여주고 있습니다. 이를 통해 우리는 projection head의 output vector에는 시각적인 정보를 많이 담고 있지 않다는 것을 확인할 수 있고, 일반적으로 downstream task은 시각적인 정보에 기반하여 해결할 수 있다는 사실로부터 encoder $f$만 사용하는 당위성을 찾을 수 있습니다.

# 5. Loss Functions and Batch Size

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-loss.png" width="70%">
</p>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-loss2.png" width="70%">
</p>
<br>

SimCLR에서는 cross-entropy 기반의 NT-Xent loss를 사용합니다. 본 논문에서는 contrastive learning에서 기존에 사용되던 다른 loss들과 달리 NT-Xent loss의 경우는 상대적으로 구분하기 어려운 negative pair에 대해 가중치를 고려할 수 있다며 loss function의 정당성을 설명하였습니다. 다른 loss의 경우에도 상대적으로 구분하기 어려운 negative sample을 고려하기 위해 semi-hard negative term을 사용하기도 하는데, 이를 다른 loss에 적용하더라도 NT-Xent loss의 성능을 능가하지 못하는 것을 볼 수 있습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-loss3.png" width="70%">
</p>
<br>

다음으로는 L2 normalization과 temperature 계수 설정의 중요성을 언급하였습니다. 위의 표를 통해 알 수 있는 점은 L2 normalization을 적용하지 않았을 경우 contrastive prediction task에서의 accuracy는 높게 나타나지만, linear evaluation의 경우에는 normalization을 적용했을 때에 비해 성능이 낮아지는 것을 확인할 수 있습니다. 따라서 우리는 L2 normalization을 적용하고 적절한 temperature 계수를 설정해줘야 합니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-batch.png" width="70%">
</p>
<br>

앞서 contrastive learning에서는 충분한 양의 negative sample이 필요하다고 언급하였습니다. 이 negative sample은 batch size에 비례하므로, SimCLR을 학습할 때 batch size를 키우는 것이 성능 향상에 효과적입니다. 또한 학습 시간이 길어질 수록 더 많은 negative sample을 학습에 반영할 수 있기 때문에 성능이 향상되는 것을 확인할 수 있습니다. 이때 training step/epochs가 커질수록 batch들은 랜덤하게 resample 되기 때문에 batch size의 영향을 덜 받게 됩니다.

# 6. Comparision with State-of-the art

논문의 저자들은 총 3가지 방법으로 실험을 진행하였습니다. 먼저, 학습된 encoder를 freeze하고 linear classifier를 추가하여 성능을 평가하였고 (linear evaluation), 다음으로 학습된 encoder와 덧붙인 linear classifier를 learnable한 상태로 학습에 사용한 dataset 중 일부만 사용하여 fine-tuning을 진행한 후 성능을 평가하였습니다 (semi-supervised learning). 마지막으로 학습된 encoder를 다른 종류의 dataset에 대해 learnable한 상태로 학습한 후 성능을 평가하였습니다 (transfer learning).

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-SOTA.png" width="70%">
</p>
<br>

먼저 linear evaluation case에서 기존의 self-supervised 방법들과 비교했을 때 SimCLR가 SOTA의 성능을 보여주었습니다. 특히 base encoder로 ResNet-50(4X)를 사용하였을 때는 supervised learning을 진행하였을 때와 성능이 비슷함을 확인할 수 있습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-SOTA2.png" width="70%">
</p>
<br>

적은 dataset에 대해 fine-tuning을 진행하였을 때도 SimCLR는 좋은 성능을 보여주고 있습니다.

<br>
<p align="center">
  <img src="{{site.baseurl}}/assets/img/Machine Learning/SimCLR-SOTA2.png" width="70%">
</p>
<br>

마지막으로 transfer learning case에서 다양한 데이터셋에 대해 학습을 진행하였을 때, supervised learning의 성능과 비슷하거나 더 좋은 성능을 보여주었습니다.