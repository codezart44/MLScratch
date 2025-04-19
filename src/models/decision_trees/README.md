# DEV NOTES
* streamlit  # Make small site displaying model and gifs of parameters being adjusted (trained)

# Decision Trees

Type of Algorithm
- Supervised Learning
- Can be used for both classification and regression
- Flexible and adaptible to many types of datasets and use cases
- Highly explainable decision making, easy to interpret

Information Gain reflects the decrease in entropy. 

$$Gain(S, A) = Entropy(S) - \sum_{v}^{A}\frac{|S_v|}{|S|}\times Entropy(S_v)$$

Gini Impurity is a measure of how often a class would be misclassified. Seek the lowest. Lower Gini Impuritiy implies more homogenous or pure distribution of classes. 

$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

Entropy is the measure of uncertainty in a random variable. 
The higher the entropy, the higher the information content. 

$$Entropy : H(S) = \sum_{v}^{A}p_v\times \log_{2}(p_v)$$


## Introduction
In this project we tackel the not so scary beast that is **Decision Trees** - An algorithm that iteratively partitions a dataset into subsets by its features (X) in pursuit of dividing the labels (y) into pure groups of classes (classification) or minimizing the variance among target values of the splits (regression). 

In this theoretical walkthough we will cover the fundamentals of Decision Trees and the underlying math used as well as some common questions about how to find these optimal splits.

The general structure of a decision tree is a branching tree-like network of nodes. Each node represents a decision point where the node splits the dataset into two (if binary) or more subgroups. The top node, also known as the root node contains the entire dataset and performs the first split, often by the most informative feature. Leaf nodes are found at the end of the tree with a final group of data. 

In general a node with successors is known as a parent node and a node with a predecessor is known as a child node.

![Illustrative Decision Tree Example](./resources/decision_tree_illustration.png)
Fig.1 - Illustrative example of Decision Tree. 

## Theory
Decision trees build on the concept of entropy - "unpredicability". The more chaos or unorder in an eventspace the less predictable outcomes become - and thus the higher the entropy becomes. 

Decision trees try to minimize the entropy (disorder) in a dataset - it does so by sorting (separating) the class labels into groups. The algorithm works by splitting the dataset by different features (X) and thresholds to partition the labels (classes). The threshold split that yields the best separation of classes is chosen, since this minimizes the entropy of the outcome. With perfect separation a state of zero entropy is reached (no mixture of classes and thus perfect order). 

Before we move onto an example, we need to cover some fundamentals of information and entropy.

### Information 
Information ($I$) is measured through the probability of an event $P(E)$. Information is measured in bits - One bit equals the information gained from the outcome of a uniform binary event space (0 or 1). Generally the Shannon Information formula is given by:
$$\text{Information} = I(p_{\omega}) = -log_2(p_{\omega}) = log_2(\frac{1}{p_{\omega}})$$
* $\Omega - \text{Event Space}$
* $\omega - \text{Event}$
* $\omega \in \Omega$
* $p_{\omega} = P(X \in \omega)$, probability of event $\omega$ happening

### Entropy
Entropy ($H$) is the expected information from a state (with some event space). Each event information is weighted by its corresponding event probability of occurring. Entropy formula given by:
$$H(S) = -\sum_{\omega \in S}p_{\omega}\times\log(p_{\omega})$$
* $S$ - State, e.g. current label distribution in dataset
* $\omega$ - Event, e.g. randomly drawing a certain label

### Information Gain
Information Gain ($IG$) is a measure of difference in entropy between two states. It is a measure of how much the entropy decreases by rearranging a system (e.g. dividing a dataset into two subsets). 
$$IG = H_{S_1} - H_{S_2}$$
$$\text{or}$$
$$IG = H_{parent} - H_{children}$$
* $S_1$ - First state
* $S_2$ - Second state

In the case of Decision Trees we partition the dataset into subsets that yield the best separation of class labels.
$$IG = H_{dataset} - \sum_{\text{subsets}}w_i\times H_i$$
* $w_i = \frac{|D_{subset_i}|}{|D_{dataset}|}$ - Subset proportion of original dataset
* $H_i$ - Entropy of subset i

This is measured after each split of the dataset, i.e. between each parent node and its children nodes in the Decision Tree (see fig.1). 

### Gini Impurity 
Gini Impurity ($GI$) is another measure of unpredicatability. Similar to entropy but with more prominent peak close to 50% probability. $1-p_i$ is the error rate, thus GI is ameasure of expected error rate. 

$$GI = \sum_i p_i(1-p_i) = 1-\sum_i{p_i}^2$$


### Reduce-Error Pruning
Reduced-Error Pruning builds on the concept of the simplest explanation (or model) being the most probable one - famously known as Occam's Razor. The idea is to avoid over-complexity, smallest accurate subtree. 

The process of:
1. Splitting the dataset into training and validation and constructing tree from training data.
2. Then iteratively removing nodes as long as validation score is not affected negatively.



Other ways of improving include bagging and bootstrap sampling. More on that in Random Forest. 

## Appendix

### Terminology
- __Root Node__: First node / decision. 
- __Branch / Internal Node__: A decision rule that splits the data. 
- __Leaf / Terminal Node__: End nodes that represent the classification.
- __Decision Boundaries__: Division made in the feature space based on the decision rules. 
- __Depth__: Longest path from root to a leaf.
- __Pruning__: Removing branches (sections) of the tree that provide little predictive power. *_Occams Razor_ 
- __Entropy__: Measure of impurity - randomness or chaos. High entropy for even distribution, low entropy for uneven distribution. 
- __Gini Impurity__: Another measure for impurity, especially used for CART algorithm for a branch.
- __Information Gain__: Difference in entropy / gini impurity before and after decision split.
- __Feature Importance__: Weight of feature for making a prediction. Which feature is most important? 

## Theory examples

### Information Examples

We can model this with a classic example of a coin toss, the event space is binary (heads H and tails T) and uniform (both sides have equal probability to land). Observing one coin toss provides us with exactly 1 bit of information. 

### Coin Toss - Shannon Information

$X \in \{H, T\}$, where X is the coin toss event.

$p_{tails} = P(X=T) = 1/2$

$I(p_{tails}) = -log_2(1/2) = log_2(2) = 1 \text{ bits}$

### Die Toss - Shannon Information

Shannon Information from a die toss. Each outcome has a certain probability and less likely events have higher infromation content. Seeing the obvious provides little to no information... The die toss outcome is modeled as a discrete random variable (rv) X following a uniform distribution (fair die).

$X \sim \mathcal{U}(1,6) = \frac{1}{6-1+1} = \frac{1}{6}$

$\Omega = \{1, 2, 3, 4, 5, 6\}$

$p_{any} = P(X = \text{Any Value}) = 1$

$$
\begin{align}
I(p_{any}) &= -log_2(p_{any}) \\
&= -log_2(1) \\
&= 0 \\
\end{align}
$$

We are guaranteed to get some number so there is no uncertainty in the event of seeing any number from a die toss. Thus the information gained on seeing any one number is 0 from a die toss. How about the case of getting an even number? 

$p_{even} = P(X = Even) = \frac{3}{6} = \frac{1}{2}$

$$
\begin{align}
I(p_{even}) &= -log_2(p_{even}) \\
&= -log_2(\frac{1}{2}) \\
&= log_2(2) \\
&= 1 \text{ bits}
\end{align}
$$

As we recall from earlier, the information gained from the outcome of a uniform binary event space is exactly 1 bit! And in this case it is binary by each outcome either being _even_ or _odd_ with the same probability. 

$p_{two} = P(X = 2) = \frac{1}{6}$

$$
\begin{align}
I(p_{two}) &= -log_2(p_{two}) \\
&= -log_2(\frac{1}{6}) \\
&= log_2(6) \\
&\approx 2.58496 \text{ bits}
\end{align}
$$

We see that the infromation has an inverse relation to the probability of the outcome - the more unlikely the event is the higher information content it holds. Rolling one specific number (2 in this case) is less likely than rolling any of the even numbers {2, 4, 6}, hence the higher information content. 


### Entropy Examples
Entropy can be regarded as the expected information content from all possible outcomes in a given state. Considering all possible outcomes $\omega_i$ from a given state and event space $\Omega$ - we compute the information content $I_{\omega}$ of each possible outcome. By then weighing each event information by its corresponding event probability and then summing we have calculated the state entropy!

Entropy is often denoted as $H$. It measures the expected amount of information that can be seen from a state given some actions.

Imagine we have a standard deck of 52 playingcards. 

Let's say we want to know the entropy for drawing a specific suit. There are four in total: Spades, Clubs, Hearts, Diamonds. These can be regarded as labels in a dataset of 52 datapoints! Drawing a specific suit is an event.

$X \sim U(1, 4) = \frac{1}{4-1+1} = \frac{1}{4}$

$$
\begin{align*}
H_{suits} &= -\sum_{suits} p(X=suit) \times \log_2(p(X=suit)) \\
&= -\sum_{i=1}^4 \frac{1}{4} \times \log_2(\frac{1}{4}) \\
&= -4 \times \frac{1}{4} \times \log_2(\frac{1}{4}) \\
&= \log_2(4) \\
&= 2
\end{align*}
$$

There is a pattern for when the distribution is uniform, allowing us to simplify to $\log_2(p)$ in the end. The only aspect affecting the final entropy when the distribution is uniform is the probability of each outcome itself, meaning for a 52 playing card deck where we compute the entropy for drawing a specific value is instead $H_{value} = \log_2(13) \approx 3.70$.

A more interesting case (and perhaps more relevant to the topic of Decision Trees) would be to understand how to compute the entropy for a non uniform dataset. When we say that we evaluate the entropy in a dataset, we actually mean the entropy of the classes (labels) - the more evenly distributed the labels are the higher the entropy is (worse separation of classes) and vice versa with perfect separation of classes the entropy would be nil. 

Let's say we have an animal dataset of 100 datapoints, containing three classes: __Monkey, Snake, Rabbit__

* 65 Monkeys
* 5  Snakes
* 30 Rabbits

The entropy in this dataset would be:

$$
\begin{align*}
H_{animals} &= -\sum_{animals} p_{animal} \times \log_2(p_{animal}) \\
&= -(p_{monkey} \times \log_2(p_{monkey}) + ... + p_{rabbit} \times \log_2(p_{rabbit})) \\
&= \frac{65}{100}\log_2(\frac{65}{100}) + \frac{5}{100}\log_2(\frac{5}{100}) + \frac{30}{100}\log_2(\frac{30}{100}) \\
&= 0.65\log(0.65) + 0.05\log(0.05) + 0.3\log(0.3) \\
&\approx 1.1412
\end{align*}
$$

This extends quite easily to any number of classes and distributions of classes.








## Questions
1. Q: __Are Decision Trees always binary trees?__
- Most decision trees are binary trees as each node has exactly two outgoing edges. Categorical features are split into groups and numerical by some threshold. For algorithms like CART (Classification and Regression Trees) the outcome (groups) are binary. For three or more classes, e.g. colors (reg, green, blue), instead of splitting by each color the splits follow (red, not red), (red & green, red & not green), (not red and green, not red & not green) etc. throughout multiple layers. This makes the tree deeper and thinner, and algorithmically easier to implement. 

2. Q: __How is the optimal threshold calculated for splitting numerical features?__
- ...


