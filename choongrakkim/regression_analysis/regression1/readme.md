

$$
\sum\limits_{i=1}^{n}(\bar{X}\bar{Y} - X_{i}Y_{i}) = \sum\limits_{i=1}^{n}(\bar{X} - X_{i})(\bar{Y} - Y_{i}) = \sum\limits_{i=1}^{n}(\bar{X} - X_{i})Y_{i} \tag{*}
$$




$$
\hat{\beta_{0}} = \bar{Y} - \bar{X} \hat{\beta_{1}} \tag{1}
$$

$$
\hat{\beta_{1}} = \frac{ \sum\limits_{i=1}^{n}(\bar{X} - X_{i})(\bar{Y} - Y_{i})}{ \sum\limits_{i=1}^{n}(\bar{X} - X_{i})(\bar{X} - X_{i})} = \frac{S_{xy}}{S_{xx}} \tag{2}
$$



### $ \mathbb{E} [\hat{\beta_{0}}]$ and $ \mathbb{E} [\hat{\beta_{1}}] $ given $X_i$



여기서, $X_i$는 given이므로 상수취급된다. $\bar{X}$와 $ S_{xx}$ 도 상수. 따라서 $w_i$도 상수. 

$$
\hat{\beta_{1}} = \sum\limits_{i=1}^{n}\frac{(\bar{X} - X_{i})}{S_{xx}}Y_{i} \\ = \sum\limits_{i=1}^{n} w_{i} Y_{i}
$$
Let $w_i = \frac{(\bar{X} - X_{i})}{S_{xx}}$
$$
\sum\limits_{i=1}^{n} w_{i} = 0 \\ \sum\limits_{i=1}^{n} w_{i}^{2} = \frac{1}{S_{xx}}
$$

$$
\mathbb{E}[\hat{\beta_{1}}] = \sum\limits_{i=1}^{n} \mathbb{E}[w_{i}Y_{i}] = \sum\limits_{i=1}^{n}w_{i}\mathbb{E}[Y_i] = \sum\limits_{i=1}^{n}w_{i}(\beta_{0 } + \beta_{1}X_i) \\ =  \beta_{0}\sum\limits_{i=1}^{n}w_{i} + \beta_{1}\sum\limits_{i=1}^{n}w_{i}X_i =\beta_{1}\sum\limits_{i=1}^{n}w_{i}X_i \\
=\beta_{1}\sum\limits_{i=1}^{n}\frac{(\bar{X} - X_{i})}{S_{xx}}X_i = \beta_{1}\sum\limits_{i=1}^{n}\frac{(\bar{X} - X_{i})X_i}{S_{xx}} = \beta_{1}\sum\limits_{i=1}^{n}\frac{(\bar{X} - X_{i})(\bar{X} - X_i)}{S_{xx}} = \beta_{1}
$$

따라서 $\hat{\beta_{1}}$은 $\beta_{1}$의 unbiased estimator이다.
$$
\mathbb{E}[\hat{\beta_{0}}] = \mathbb{E}[\bar{Y} - \bar{X} \hat{\beta_{1}}] = \mathbb{E}[\bar{Y}] - \bar{X} \mathbb{E}[\hat{\beta_{1}}] = \mathbb{E}[ \frac{1}{N}\sum\limits_{i=1}^{n} \beta_0 + \beta_1 X_i + \epsilon_i] -\bar{X}\beta_{1} \\ = \mathbb{E}[ \frac{1}{N}\sum\limits_{i=1}^{n} \beta_0 + \beta_1 X_i]+  \mathbb{E}[\frac{1}{N}\sum\limits_{i=1}^{n} \epsilon_i] -\bar{X}\beta_{1} \\ = \beta_{0} + \bar{X}\beta_{1} + \mathbb{E}[\epsilon]-\bar{X}\beta_{1}
\\ = \beta_{0}
$$
$\hat{\beta_{0}}$도 $\beta_{0}$의 unbiased estimator이다.


### $ \mathbb{Var} [\hat{\beta_{0}}]$ and $ \mathbb{Var} [\hat{\beta_{1}}] $ given $X_i$



여기서, $Y_i = \beta_{0} + \beta_{1} X_i + \epsilon_i$의 randomness는 iid $\epsilon$에 의해 주어진다. 따라서 각각의 $Y_i$끼리는 independent 이다.
$$
\mathbb{Var}[\hat{\beta_{1}}] = \sum\limits_{i=1}^{n} \mathbb{Var}[w_{i}Y_{i}] = \sum\limits_{i=1}^{n}w_{i}^{2}\mathbb{Var}[Y_i] = \sum\limits_{i=1}^{n}w_{i}^{2} \sigma_{\epsilon}^{2} = \frac{\sigma_{\epsilon}^{2}}{S_{xx}}
$$






$$
\mathbb{Var}[A + B] = \mathbb{Var}[A] + \mathbb{Var}[B] + 2 \mathbb{Cov}(A, B) \tag{**}
$$
$$
\mathbb{Var}[A - B] = \mathbb{Var}[A] + \mathbb{Var}[B] - 2 \mathbb{Cov}(A, B) \tag{**}
$$

$$
\mathbb{Cov}(A, B) = \mathbb{Cov}(B, A) \tag{***}
$$

$$
\mathbb{Cov}(aA, B) = a\mathbb{Cov}(A, B)   \tag{****}
$$



$$
\mathbb{Var}[\hat{\beta_{0}}] = \mathbb{Var}[\bar{Y} -\hat{\beta_{1}}\bar{X}] = \mathbb{Var}[\bar{Y}] + \mathbb{Var}[\hat{\beta_{1}} \bar{X}] - 2 \mathbb{Cov}(\bar{Y}, \hat{\beta_{1}} \bar{X}) \\
= \mathbb{Var}[\bar{Y}] + \bar{X}^{2} \mathbb{Var}[\hat{\beta_{1}}] - 2  \bar{X} \mathbb{Cov}(\bar{Y}, \hat{\beta_{1}}) \\
= \frac{\sigma_{\epsilon}^{2}}{N} + \frac{\sigma_{\epsilon}^{2}}{S_{xx}} \bar{X}^{2} - 2  \bar{X} \mathbb{Cov}(\bar{Y}, \hat{\beta_{1}}) = \sigma_{\epsilon}^{2}  (\frac{1}{N} + \frac{\bar{X}^{2}}{S_{xx}})
$$

여기서 $\mathbb{Cov}(\bar{Y}, \hat{\beta_{1}}) = 0$이다. 증명은 다음과같다.
$$
\mathbb{Cov}(\bar{Y}, \hat{\beta_{1}}) = \mathbb{Cov}(\frac{1}{N}\sum\limits_{i=1}^{n} Y_i, \sum\limits_{i=1}^{n} w_{i} Y_{i}) = \mathbb{Cov}(\frac{1}{N}1^{t}Y, W^{T}Y) = \frac{1}{N}1^{t} \mathbb{Cov}(Y, Y)W
$$
여기서 $Y_i$는 독립이므로 $\mathbb{Cov}(Y, Y)$는 $\sigma_{\epsilon}^{2} I$이다. 
$$
\frac{1}{N}1^{t} \mathbb{Cov}(Y, Y)W = \frac{\sigma_{\epsilon}^{2}}{N}1^{t} W = \frac{\sigma_{\epsilon}^{2}}{N}\sum\limits_{i=1}^{n} w_{i} = 0
$$




### properties of residual

Observed value $Y_i$와 fitted value $\hat{Y_i}$의 차이를 residual 이라고 한다.

$Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$

$\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1} X_i$

$e_i = Y_i - \hat{Y_i} $ = i-th residual




$$
\sum\limits_{i=1}^{n} e_{i} = \sum\limits_{i=1}^{n} Y_i - \hat{Y_i}  \\ = \sum\limits_{i=1}^{n}( Y_i - \hat{\beta_0 }- \hat{\beta_1} \hat{X_i}) = 0 \tag{a}
$$

위의 식은 손실함수에서 $\beta_0$ 편미분 = 0한 식이다. 


$$
\sum\limits_{i=1}^{n} X_i e_{i} = \sum\limits_{i=1}^{n} X_i (Y_i - \hat{Y_i}) = 0 \tag{b}
$$

위의 식은 손실함수에서 $\beta_1$ 편미분 = 0한 식이다. 



$$
\sum\limits_{i=1}^{n} \hat{Y_i} e_{i} = \sum\limits_{i=1}^{n} (\hat{\beta_0} + \hat{\beta_1} X_i)e_i = 0 \tag{c}
$$
(a), (b)를 통해 0임을 확인할 수 있다.







### evaluate the fitted value at $\bar{X}$

$$
\hat{\beta_0} + \hat{\beta_1} \bar{X} =\frac{1}{N} \sum\limits_{i=1}^{n} \hat{\beta_0} + \hat{\beta_1} X_i =\frac{1}{N} \sum\limits_{i=1}^{n} \hat{Y_i} = \bar{Y}
$$

fitted model이 항상 좌표 $\bar{X}, \bar{Y}$를 지난다.

(a)를 이용해 확인할 수있다. 





### Gauss-Markov  theorem

Least square estimator는 Unbiased Linear estimator중에 가장 variance가 작다.(Best이다. = Best Linear Unbiased estimator)



#### $\hat{\beta_{1}}$ 증명

$$
\mathbb{Var}(\hat{\beta_{1}}) \le \mathbb{Var}(\hat{\beta_{1}^{*}})
$$

를 보이면 된다. 



##### Linearity Condition

$$
\hat{\beta_1}^{*} = \sum\limits_{i=1}^{n} c_i Y_i \tag{condition-1}
$$
여기서 $c_i$ 는임의의 상수. 

$$
\hat{\beta_1}^{*} = \sum\limits_{i=1}^{n} (w_i + d_i) Y_i
$$
여기서 $w_i= \frac{X_i - \bar{X}}{S_{xx}}$이고, $d_i$ = $c_i - w_i$



##### Unbiased estimator condition

$$
\mathbb{E}[\hat{\beta_{1}}^{*}]= \beta_1 \tag{condition-2}
$$

$$
\mathbb{E}[\hat{\beta_{1}}^{*}] = \mathbb{E}[\sum\limits_{i=1}^{n} (w_i + d_i)(\beta_0 + \beta_1 X_i + \epsilon_i)]
$$

여기서, $\epsilon$ 빼고 다 상수이므로 expectation에서 빠져나올 수 있다.
$$
=\sum\limits_{i=1}^{n} (w_i + d_i)\beta_0 + \sum\limits_{i=1}^{n} (w_i + d_i)\beta_1 X_i
$$

$$
=\beta_0\sum\limits_{i=1}^{n} d_i + \beta_1\sum\limits_{i=1}^{n} d_i X_i +\beta_1\sum\limits_{i=1}^{n} w_i X_i
$$

인데 $\sum\limits_{i=1}^{n} w_i X_i = 1$ 이다. 따라서

condition-1, 2에 의해 $\hat{\beta_1}^{*}$의 다음 두 제약을 얻을 수 있다.
$$
\sum\limits_{i=1}^{n} d_i = 0 \tag{r1}
$$
$$
\sum\limits_{i=1}^{n} d_i X_i = 0 \tag{r2}
$$




$$
\mathbb{Var}[\hat{\beta_1}^{*}]=\mathbb{Var}[\sum\limits_{i=1}^{n} (w_i + d_i) Y_i]
$$
$Y_i$ 들은 independent 이므로
$$
=\sum\limits_{i=1}^{n}(w_i + d_i) ^{2}\mathbb{Var}[Y_i] = \sum\limits_{i=1}^{n}(w_i + d_i) ^{2} \sigma_{\epsilon}^{2}
$$

$$
=\sigma_{\epsilon}^{2}(\sum\limits_{i=1}^{n} w_i^{2} + 2\sum\limits_{i=1}^{n} w_i d_i + \sum\limits_{i=1}^{n} d_i^{2})
$$

한편 
$$
\mathbb{Var}[\hat{\beta_1}]= \sigma_{\epsilon}^{2} \sum\limits_{i=1}^{n}w_{i}^{2}
$$
이므로
$$
\mathbb{Var}[\hat{\beta_1}^{*}] =\mathbb{Var}[\hat{\beta_1}] + 2 \sigma_{\epsilon}^{2} \sum\limits_{i=1}^{n} w_i d_i + \sigma_{\epsilon}^{2} \sum\limits_{i=1}^{n} d_i^{2}
$$
이다. 근데 
$$
\sum\limits_{i=1}^{n} w_i d_i = \frac{1}{S_{xx}} \sum\limits_{i=1}^{n}(X_i - \bar{X})d_i = \frac{1}{S_{xx}}( \sum\limits_{i=1}^{n}X_i d_i -\bar{X} \sum\limits_{i=1}^{n}d_i) = 0
$$
이므로 
$$
\mathbb{Var}[\hat{\beta_1}^{*}] =\mathbb{Var}[\hat{\beta_1}] + \sigma_{\epsilon}^{2} \sum\limits_{i=1}^{n} d_i^{2} \ge \mathbb{Var}[\hat{\beta_1}]
$$
이된다.
