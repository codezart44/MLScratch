# Logistic Regression
___

## Introduction
$\mathbf{X}$ is a matrix of $m$ samples and $n$ features, $w$ is a vector of $n$ weights and $b$ a single scalar bias. 

$$              
\begin{equation}
\mathbf{z} = \mathbf{X} \cdot \mathbf{w} + b 
\end{equation}
$$

The sigmoid function maps the real numberline to the open interval 0 to 1.

$\sigma: \mathbb{R} \mapsto (0, 1)$

$$
\begin{equation}
\sigma{(\mathbf{z})} = \frac{1}{1+e^{-\mathbf{z}}} 
\end{equation}
$$

The sigmoid functions can be treated as a mapping of each value $z_i$ to  probability $p_i$. 


Calculating error with cross entropy function:

The cross entropy loss function comes from the MLE of estimating w & b that maximizes the likelihood of observed data. 

Given some input xi the predicted probability $P(y=1|x_i)$ for the positive class is given by:

$$
\begin{equation}
p_i = P(y=1|\mathbf{x_i}) = \frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w}+b)}}, \hspace{3mm} \text{0 < i < m}
\end{equation}
$$

## Further explanation
#### Likelihood vs probability <br>
_Probability_ tells us the chance of observing some data with a given model. Can give the chance of a specific event occuring or observing values outside or inside of ranges of a probability distribution.

_Likelihood_ tells us the "probability density" of a probability density function (PDF) at a certain point x. This is litterally the function value (height of the graph) at the point x. This distribution is usually dependent on some amount of parameters, for example the mean $\mu$ and standarddeviation $\sigma$ as in the case of the normal distribution. 

To further clarify what is meant by likelihood... From some relationship or phenonmenon we may have randomly observed some datapoints $\set{x_1, x_2, ...}$. To model this relationship we assume a distribution with a PDF with for example the parameter $\theta$. Since the data was randomly observed (selected) we assume the distribution of observed datapoints $\set{x_1, x_2, ...}$ is representative of the true relationship (more datapoints means less variance and more accurate representation). To find a good approximation for the true relationship we want to adjust the parameter $\theta$ for the PDF to explain the observed data, representing the true relationship, as good as possible. This means finding the $\theta$ that maximizes the likelihood of observing exactly these datapoints $\set{x_1, x_2, ...}$ together (i.e. joint likelihood) under the assumed PDF. This method is called maximum likelihood estimation, MLE for short. 

In logistic regression we do not have a PDF in the common sense. The sigmoid function instead models a conditional probability and not probability density over a stochastic variable. 

$$
\begin{equation}
p_i = P(y=1|\mathbf{x_i}) = \frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w}+b)}}, \hspace{3mm} \text{0 < i < m}
\end{equation}
$$

The logistic (sigmoid) function tells us the probability of seeing the positive label $y_i = 1$ for a given datapoint $\mathbf{x_i}$. This is not a PDF since it does not represent a distribution of probability, integrating the logistic function over the real numberline $\mathbb{R}$ does not yield a cumulative probability of 1 as the CDF (cumulative distribution function) should and does. Meaning no, the logistic function is not a PDF.

Instead we want to maximize the joint likelihood of observing the binary labels $\mathbf{y} = (y_1, y_2, \ldots)$ given the data  $\mathbf{X} = (\mathbf{x_1}, \mathbf{x_2}, \ldots)$. Assuming $z_i = \mathbf{x_i} \cdot \mathbf{w} + b$ is a strength measure for the probability of observing $y_i = 1$ we can model this relationship fairly accurate with the logistic (sigmoid) function, whereas large negative values of $z_i$ yields low probability of observing $y_i = 1$ and large positive values of $z_i$ yields high probability of observing $y_i = 1$ and a small range of high uncertainty for $z_i$ close to zero (around the inflection point of the curve). The logistic function can be seen as a "Probability Intensity Function", with probability of $y_i=1$ increasing as $z_i$ does. Though note that "probability intensity function" is not common terminology within machine learning or statistics. 

The fit of the model to each data pair of $\mathbf{x_i}$ and $y_i$ is given by the likelihood of observing the correct label $y_i$ with our current parameter values for $\mathbf{w}$ and $b$. 

$$
\begin{equation}
l_i = L(\mathbf{w}, b; y_i, \mathbf{x_i})
\end{equation}
$$

$$
\begin{equation}
l_i = p_i^{y_i} \times (1-p_i)^{1-y_i}, \hspace{3mm} y_i \in \set{0, 1}
\end{equation}
$$

and the overall model fit to the whole dataset is given by the joint likelihood 

$$
\begin{equation}
L = L(\mathbf{w}, b; \mathbf{y}, \mathbf{X}) = l_1 \times l_2 \times \ldots \times l_m
\end{equation}
$$

$$
\begin{equation}
L = \prod_{i=1}^{m}p_i^{y_i} \times (1-p_i)^{1-y_i}, \hspace{3mm} y_i \in \set{0, 1}
\end{equation}
$$

To give some intuition for this extension of the logistic function... We want to maximize the joint likelihood no matter $y_i = 1$ or $y_i = 0$ meaning we need to complement the original logistic function (only accounting for $y_i = 1$) by adding a factor that measures the probability of observing $y_i = 0$. If $p_i$ represents the probability of observing $y_i = 1$ then the probability of observing the other outcome $y_i = 0$ is neatly given by $(1-p_i)$. These factors are then raised to $y_i$ and $(1-y_i)$ respectively making only the probability of the current label's value ($p_i$ if $y_i = 1$ and $(1-p_i)$ if $y_i = 0$) accounted for in the final likelihood measure. 

$$
p_i = P(y_i=1|\mathbf{x_i})
$$

$$
(1-p_i) = P(y_i=0|\mathbf{x_i})
$$


## Maximum Likelihood Estimation - MLE
For a single observation $x_i$ the likelihood of seeing $y_i = 1$ is given by:

$$
\begin{equation}
l_i = p_i^{y_i} \times (1-p_i)^{1-y_i}
\end{equation}
$$


For $m$ observations in $\mathbf{X}$ the joint likelihood of seeing the outcome $\mathbf{y}$, where each $y_i \in \set{0, 1}$, is given by the product of the individual likelihoods for each $y_i$:

$$
\begin{equation}
L = \prod_{i=1}^{m}p_i^{y_i} \times (1-p_i)^{1-y_i}
\end{equation}
$$

Since we seek the MLE (maximum likelihood estimation) we want to find the global maximum of this function.

Note that for $f'(x) = 0$, signifying an extreme point, $\frac{d}{dx} \log(f(x)) = \frac{f'(x)}{f(x)} = 0$. Maximizing $L$ or $\log(L)$ yields the same estimates since the logarithm is a monotonically increasing function, that is if $b > a$ then $\log(b) > \log(a)$ thus preserving the characteristics of maxima and minima. 

As an additional benefit the logarithms transforms the product into a sum simplifying the differentiation of $L$. It also turns small joint probabilities into large negative sums avoiding computational underflow.

$$
\begin{equation}
\log(L) = \sum_{i=1}^{m} y_i \times \log(p_i) + (1-y_i) \times \log(1-p_i)
\end{equation}
$$

//cross entropy

...according to the log laws <br>
i) $\hspace{2.1cm} \log(A \times B) = \log(A) + \log(B)$

ii) $\hspace{2cm} \log(A^b) = b \times \log(A)$



Instead of trying to maximize the negative value os the sum of log-likelihoos we instead try to minimize the positive value of the negative log-likelihood turning it into a loss function. This is a standard in machine learning optimization, that you would want to minimize the loss function. Dividing by $m$ gives the mean of negative log-likelihoods. This measure is often abbreviated as the NLL - negative log-likelihood.

$$
\begin{equation}
-\log(L)/m = -\frac{1}{m}\sum_{i=1}^{m} y_i \times \log(p_i) + (1-y_i) \times \log(1-p_i)
\end{equation}
$$

$$
\begin{equation}
NLL = -\frac{1}{m}\sum_{i=1}^{m} y_i \times \log(p_i) + (1-y_i) \times \log(1-p_i)
\end{equation}
$$



The NLL function quantifies how well the sigmoid curve conforms to the label values $\mathbf{y}$, where values close to zero means good fit (conformity) and large values means bad fit (a lot of loss). <br>


#### Example low loss, good approximation

• $\hspace{1cm} y_i = 0, \hspace{2mm} p(y_i=1|x_i) = 0.05$

• $\hspace{1cm} nll_i = -\log(l_i) = -(0 \times \log(0.05) + (1-0) \times \log(1-0.05)) = -\log(0.05) \approx 0.022$

#### Example medium loss, decent approximation

• $ \hspace{1cm} y_i = 1, \hspace{2mm} p(y_i=1|x_i) = 0.8$

• $\hspace{1cm} nll_i = -\log(l_i) = -(1 \times \log(0.8) + (1-1) \times \log(1-0.8)) = -\log(0.8) \approx 0.097$

#### Example high loss, bad approximation

• $\hspace{1cm} y_i = 1, \hspace{2mm} p(y_i=1|x_i) = 0.05 $

• $\hspace{1cm} nll_i = -\log(l_i) = -(1 \times \log(0.05) + (1-1) \times \log(1-0.05)) = -\log(0.05) \approx 1.30$


We want the negative log-likelihood loss to be as close as possible to zero and thus try adjust model parameters $b$ & $w$ to minimize the NLL function value.



## Finding partial differentials

$\frac{\partial{f}}{\partial{b}}$ & $\frac{\partial{f}}{\partial{w}}$

Negative Log Likelihood - NLL (loss)

$$
NLL = -\frac{1}{m}\sum_{i=1}^{m} y_i \times \log(p_i) + (1-y_i) \times \log(1-p_i)
$$

// rewriting the function 

{

(1) $\hspace{2cm} nll_i = y_i \times \log(p_i) + (1-y_i) \times \log(1-p_i)$

(2) $\hspace{2cm} nll_i = y_i \times \log(\frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) + (1-y_i) \times \log(1-\frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

(3) $\hspace{2cm} nll_i = y_i \times \log(\frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) + (1-y_i) \times \log(\frac{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}} - \frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

(4) $\hspace{2cm} nll_i = y_i \times \log(\frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) + (1-y_i) \times \log(\frac{e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

(5) $\hspace{2cm} nll_i = y_i \times \log(\frac{1}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) + (1-y_i) \times \log(e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}) - (1-y_i) \times \log({1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

(6) $\hspace{2cm} nll_i = -y_i \times \log(1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}) + (y_i-1) \times (\mathbf{x_i} \cdot \mathbf{w} + b) - \log({1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) + y_i \times \log({1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

(7) $\hspace{2cm} nll_i = (y_i-1) \times (\mathbf{x_i} \cdot \mathbf{w} + b) - \log({1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}) $

// rewriting the function end

$$
\begin{equation}
NLL = f(b,w) = -\frac{1}{m}\sum_{i=1}^{m} (y_i-1) \times (\mathbf{x_i} \cdot \mathbf{w} + b) - \log({1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}})
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b} = -\frac{1}{m}\sum_{i=1}^{m} (y_i-1) + \frac{e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial w} = -\frac{1}{m}\sum_{i=1}^{m} \mathbf{x_i} \times \left[(y_i-1) + \frac{e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}}{1+e^{-(\mathbf{x_i} \cdot \mathbf{w} + b)}} \right]
\end{equation}
$$

## Scaling
#### y data
Not needed for $\mathbf{y}$ data since $\forall y_i \in \mathbf{y}, \hspace{2mm} y_i \in \set{0, 1}$ meaning $\mathbf{y}$ only consists of the binary values 0 or 1.<br>

#### X data
Usually beneficial and sometimes necessary for $\mathbf{X}$ data. Need to align features and avoid exploding- / vanishing gradients.


## Notes (Old)
___
Q: __*Why the product of likelihoods and not the sum of likelihoods?*__

$$
\begin{equation}
L = \prod_{i=1}^{m}p_i^{y_i} \times (1-p_i)^{1-y_i}
\end{equation}
$$

$$
\begin{equation}
? \hspace{5mm} L = \sum_{i=1}^{m}p_i^{y_i} \times (1-p_i)^{1-y_i} \hspace{5mm} ?
\end{equation}
$$

A: The reason lies in the nature of probability, likelihood and the difference between additon (sums) and multiplication (products). When we train the model we seek a generalized solution to the training data where all of the datapoints are evaluated as joint probability. 

If A has a 50% chance of occuring and B has a 60% chance of occuring the joint likelihood of A & B happening is given by the product of each individual probability (given events A and B are independet of eachother).

$ P(A \cap B) = P(A) \times P(B) = 0.5 \times 0.6 = 0.3 = 30\% $

Adding the probabilities would not make sense in this case...

$ P(A \cap B) = P(A) + P(B) = 0.5 + 0.6 = 1.1 = 110\% ?$
___

Q2: __*Could we still not use the sum of likelihoods to optimize out model in a practical sense?*__

The model is not fit to individual datapoints but rather the whole dataset. Minimizing the sum of likelihoods poses problems since large likelihoods can compensate for small likelihoods. This is not the case for maultiplication which adheres to the nature of probabilities. 


$ \text{Example of sum vs. product} \\$
$ \text{Case 1:} \\$
$ P(A)=0.5, \hspace{2mm} P(B)=0.5 \\$
$ \text{Sum: } L(A) + l(B) = 0.5 + 0.5 = \\$


$ \text{Example high loss, bad approximation} \\$
$ y_i = 1, \hspace{2mm} p(y_i=1|x_i) = 0.05 \\$
$ nll_i = -\log(l_i) = -(1 \times \log(0.05) + (1-1) \times \log(1-0.05)) = -\log(0.05) \approx 1.30$

P(A) = 0.5, P(B) = 0.5
Addition:               P(A&B) = 0.5+0.5 = 1.0
Multiplication:         P(A&B) = 0.5*0.5 = 0.25

P(A) = 0.05, P(B) = 0.95
Addition:               P(A&B) = 0.05+0.95 = 1.0
Multiplication:         P(A&B) = 0.05*0.95 < 0.05

This creates the need to generalize the model to have similar fit to all datapoints instead of extremely well fit to some and terrible to other.

With addition large likelihoods can compensate for small likelihoods as seen in the example above where even though the likelihood of A is only 0.05 the large likelihood of B at 0.95 weighs up for that. This causes to model to not generalize well but rather increases variance as fitting well to some datapoints can compoensate for bad fits to other datapoints. That is not the case for multiplication where the maximum joint likelihood is reached when all likelihoods are close to equal. 

Multiplying many small likelihoods together poses other problems however, a small joint likelihood (product) and difficulty differentiating. To compensate for miniscule joint likelihoods we use the log of the product which firstly turns it into a sum (facilitating differentiation) and secondly turns small decimal likelihoods into large negative values. 
         N
log(L) = ∑(log(p^yi) + log((1-p)^(1-yi)))
        i=1

By multiplying by negative one (-1) and dividing by the number of samples the negative logs turns positive and the optimization problem turns into a problem of minmizing the average of the joint negative log-likelihood (NLL) 
                 N
-log(L)/m = -1/m*∑(log(p^yi) + log((1-p)^(1-yi)))
                i=1

This aligns with the standard of machine learning optimization problems which consist of minimizing the loss/cost of an accuracy function of some kind. 