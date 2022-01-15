# RetinaNet

"Focal Loss for Dense Object Detection" 논문 기반으로 공부 + 블로그 참고. <br>
기존의 one-stage detection의 foreground와 background의 class imbalance문제의 해결책 제시. <br>
retinaNet은 기존의 model들을 backbone으로 focal loss를 적용한 형태이기에 focal loss를 중점으로 공부. <br>
###### ( 기존 내용도 차근차근 올릴 예정...ㅎ )

<br>

---

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

---

### Focal Loss
덜 분류된 것들에게는 많은 loss를 주고 잘 분류된 것들에게는 적은 loss를 주는 것이 주된 idea

<br>

#### - Cross Entropy Loss
잘못 예측한 경우 패널티를 부여하는 느낌. <br>

<img width="238" alt="스크린샷 2022-01-16 오전 12 43 29" src="https://user-images.githubusercontent.com/55525705/149627908-4a0a3449-cb13-4f49-9a2e-5cf4ecd2cc1c.png">

cross entropy의 식은 위와 같음. (여기서 pt는 object일 확률) 식만 보면 문제가 없어보이지만 위쪽에 기재하였던 2번째 문제 발생. <br>
foreground와 background가 같은 값의 loss를 갖고 있는 상태에서 계속 누적되면 두 class의 비율이 다르기 때문에 background의 loss가 늘어날 것이고 이는 곧 더 많은 학습을 하게 됨을 의미.

<br>

#### - Balanced Cross Entropy Loss
cross entropy loss는 foreground와 background의 비율이 다른 것( = class imbalance )을 적용하지 못해서 발생하는 문제로 추가로 weight를 곱해주는 방식. <br>
각 class 수의 역수를 weight로 loss에 곱해주면 class수가 많은 background같은 경우 loss가 작게 반영될 것. <br>

<img width="168" alt="스크린샷 2022-01-16 오전 12 51 45" src="https://user-images.githubusercontent.com/55525705/149628152-fe194a03-7c2a-4778-adcf-8cac0e7365db.png">

balanced cross entropy식은 위와 같음. 그러나, 이 방식에도 문제가 존재. <br>
class의 수가 많으면 무조건 easy negative로 판단한다는 것. ( = easy, hard negative 구분이 불가능 )

<br>

#### - Focal Loss
위의 두 loss들의 문제점을 개선하고자 만들어짐. <br>

<img width="450" src="https://user-images.githubusercontent.com/55525705/149629266-7eb003a7-ff15-440b-af5e-00ee2faa64de.png">

(1-pt)^r을 통해서 balanced cross entropy문제를 개선. ( = easy, hard negative를 구분하는 부분 ) <br>
기존의 balanced cross entropy의 문제를 반복하지 않기 위해서 r 값을 적절하게 조절해주는 것이 중요. (논문에서는 2 사용) <br>


<img width="200" src="https://user-images.githubusercontent.com/55525705/149625334-d496b72d-4216-401a-8dad-afb79d68b95f.png">

위 그림에서 빨간 선이 보통의 Binary Cross Entropy로 x가 1이면 분류가 잘 된 것으로, 학습을 진행할수록 loss값이 작아지는 것이 이상적임. <br>
파란색 선이 focal loss로 -(1-pt)^r * log(pt)의 개형으로, x의 값이 커질수록 loss 값이 원래의 loss ( = 빨간색) 보다 작아짐. 
 ###### => easy negative에 대해서는 loss를 조금줘서 영향력을 낮춤
반대로 hard negative ( = x가 0에 가까운 sample )에 대해서는 기존 loss 보다 큰 loss를 줌.
 ###### => hard negative에 대한 비중을 늘림

<br>

위의 단순 -(1-pt)^r * log(pt)뿐만 아니라 정답일때와 아닐때 곱해주는 값을 다르게 해주기 위해서 balanced Cross Entropy라 불리는 것을 더 사용해서 최종 Focal Loss 만들게 됨.

<img width="300" src="https://user-images.githubusercontent.com/55525705/149625789-58186aa4-f8ec-46c8-8ab5-817bc65db211.png">

<br>

#### - Initialization
Focal Loss를 사용할 때는 초기화 작업이 중요하다고 함. <br>
그런데 이 초기화 작업이 기존의 binary classification 과는 조금 다름. <br>
 - 기존 : 보통 initialization을 할 때 두 label의 확률을 0.5로 같게 초기화 함
 - focal loss : rare class 인 것처럼 초기화

__why?__ 기존의 label의 확률을 0.5, 0.5로 만들어주면 class imbalance로 인해서 우세한 class가 생겨버리게 됨. 이때, rare class인 것처럼 초기화를 0.01로 작은 값으로 해주게 되면 class imbalance로 인해 생기는 초기화 문제를 피할 수 있음.
<br>
__how?__ sigmod이후 값을 0.01로 고정시키고 확인

<img width="300" src="https://user-images.githubusercontent.com/55525705/149626879-f0e8dd32-c7b9-499c-b04d-c78348f9db9c.png">

위 그림과 같은 과정을 통해서 bias초기화 가능하고 wieght 는 N(0, 0.01) 에서 sampling 을 하여 초기화를 진행. <br>
convolution weight 가 0 에 가깝고 bias 가 위에서 초기화한 식의 값이 되므로, bias 만 남게되어 초기의 sigmoid 이후의 값을 (0.01) 로 제한 가능

 ###### 요 부분은 더 공부가 필요! :sob:

