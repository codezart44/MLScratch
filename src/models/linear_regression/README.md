## Linear Regression implemented from scratch
___

- The point of this project is to understand the math behind linear regression. 
- Linear regression finds a best fit line through a set of datapoints and uses that line to evaluate (predict) unseen data. 
- It is assumed that the dataset follows a linear pattern.


### Ordinary Least Squares (OLS)
___
* Find closest approximated solution β' to X•β = y
* Projecting y down on Col X and find a solution for X•β' = ŷ
```
X.T • Xβ' = X.T • y
β' = (X.T • X)^-1 • (X.T • y)
```
//note requires X to have linearly independent columns to be invertible (IMT)


### Gradient Descent (GD)
___
* Minimize MSE (cost function) 
* Calculating pertial derivatives of cost funtion with respect to b & w
```
                      n
MSE = f(b, w) = 1/N * ∑(yi - (w*xi + b))^2
                     i=1

# repeat n times:
            N
db = -2/N * ∑(y - (X•w + b))
           i=1
dw = -2/N * X.T • (y - (X•w + b))

b = b - ɑ * db
w = b - ɑ * dw
```
//note sensitive to learning rate and initial values for bias & weights. Requires scaling.


### Scaling & rescaling 
___
Standardizing data
```
µx, µy = mean(X), mean(y)
σx, σy = std(X), std(y) 
Xstd = (X - µx)/σx
ystd = (y - µy)/σy

# Xrbu Robust scaling
# yrbu

# Xmmx minmax scaling
# ymmx

# Xmab maxabs scaling
# ymab
```

Rescaling weights and bias back to original scale
```
from the scaling we know:

Xstd = (X - µx)/σx
ystd = (y - µy)/σy

And in the approximation of ystd we have:
ystd = w * Xstd + b
(y - µy)/σy = w*(X - µx)/σx + b
y - µy = w*(X - µx)σy/σx + b*σy
y - µy = w*σy/σx*X - w*µx*σy/σx + b*σy
y = w*σy/σx*X - w*µx*σy/σx + b*σy + µy
y = (w*σy/σx)X + (b*σy + µy - w*µx*σy/σx)

and since y = w_orig * X + b_orig we have:

w_orig * X + b_orig = (w*σy/σx)X + (b*σy + µy - w*µx*σy/σx)

and comparing term by term we find:

w_orig = w * σy/σx
b_orig = b * σy + µy - w * µx * σy/σx
```

### Notes
___
![linreg_note_p1](./notes/linreg_note_p1.jpg)
p.1
___
![linreg_note_p2](./notes/linreg_note_p2.jpg)
p.2
___
![linreg_note_p3](./notes/linreg_note_p3.jpg)
p.3
___
![linreg_note_p4](./notes/linreg_note_p4.jpg)
p.4
