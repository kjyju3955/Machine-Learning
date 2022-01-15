# RetinaNet
---
"Focal Loss for Dense Object Detection" 논문 기반으로 공부 + 블로그 참고. <br>
기존의 one-stage detection의 foreground와 background의 class imbalance문제의 해결책 제시.

<br>

### foreground, background class imbalance
여러개의 anchor중에서 object를 포함하고 있는 anchor ( = foreground) 개수보다 배경을 담고 있는 anchor ( = background) 의 수가 훨씬 더 많음. 

<img width="780" alt="스크린샷 2022-01-15 오후 10 44 09" src="https://user-images.githubusercontent.com/55525705/149623844-d093b205-b028-420e-a75b-53c3c448f189.png">

 - 오른쪽의 몇 개의 object anchor를 위해서 왼쪽처럼 무수히 많은 anchor들이 사용됨.

<br>

#### - 문제
##### 1. easy negative ( = 배경 ) 이 너무 많기 때문에 object를 찾는데에 비효율적
##### 2. easy negative 각각의 loss는 작지만, easy negative의 개수가 많아 전체 anchor에서 비중이 크기 때문에 easy negative의 영향이 커져 성능이 떨어짐

<br>

#### - 위의 문제를 해결하는 방법
- YOLOv2 : object인지 배경인지 판단하는 기준을 conf term과 class score의 conditional probability로 정의하여 그 둘의 곱으로 두어서 학습
- SSD : hard negative minig 방식 사용
- ***RetinaNet : Focal Loss 사용*** 

    ###### YOLOv2와 SSD부분은 추후 공부할 예정
    
<br>

### Focal Loss
덜 분류된 것들에게는 많은 loss를 주고 잘 분류된 것들에게는 적은 loss를 주는 것이 주된 idea

<img width="200" src="https://user-images.githubusercontent.com/55525705/149625334-d496b72d-4216-401a-8dad-afb79d68b95f.png">

위 그림에서 빨간 선이 보통의 Binary Cross Entropy로 x가 1이면 분류가 잘 된 것으로, 학습을 진행할수록 loss값이 작아지는 것이 이상적임. <br>
파란색 선이 focal loss로 -(1-x)^2 * log(x)의 개형으로, x의 값이 커질수록 loss 값이 원래의 loss ( = 빨간색) 보다 작아짐. 
 ###### => easy negative에 대해서는 loss를 조금줘서 영향력을 낮춤
반대로 hard negative ( = x가 0에 가까운 sample )에 대해서는 기존 loss 보다 큰 loss를 줌.
 ###### => hard negative에 대한 비중을 늘림

<br>

위의 단순 -(1-x)^2 * log(x)뿐만 아니라 정답일때와 아닐때 곱해주는 값을 다르게 해주기 위해서 balanced Cross Entropy라 불리는 것을 더 사용해서 최종 Focal Loss 만들게 됨.

<img width="300" src="https://user-images.githubusercontent.com/55525705/149625789-58186aa4-f8ec-46c8-8ab5-817bc65db211.png">

