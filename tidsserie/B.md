# B.  Modeling and Validation for El-Geneina

We are tasked to model El-Geneinas vegetation (NDVI data), both with and without precipitation as an external signal. And to find out if the use of the precipitation improves our predictions. We can also not use any recursive models until task C.

We split the data into model (75%), validation (15%) and test (10%). As we didn't have much data to go by, just 648 samples, we wanted to have as much used for modelling, although it makes model validation and test less accurate. 

## ARMA model

### Key motivational steps to obtain model

Data looks roughly stationary and normal so no transform needed.

![Skärmavbild 2024-01-06 kl. 13.34.36](/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-06 kl. 13.34.36.png)

ACF below shows season of 36, so removing it, by fitting A polynomial convolved with AR(36) .

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-07 kl. 13.58.40.png" alt="Skärmavbild 2024-01-07 kl. 13.58.40" style="zoom:20%;" />

We do the standard model order estimation procedure (by analyzing ACF & PACF, and fitting parameters with PEM). Below are the residuals of final fit shown.

<img src="/Users/edgar/Desktop/Skärmavbild 2024-01-09 kl. 13.47.06.png" alt="Skärmavbild 2024-01-06 kl. 13.38.10" style="zoom:50%;" />

Residuals look white and Monti-test deems them to be white also as 8.57 < 31.41. They aren’t normal though, but they look normal enough and as 8.57 is a lot less than 31.41, the residuals shoud be white.

```
Discrete-time ARMA model: A(z)y(t) = C(z)e(t)

A(z) = 1 - 0.7076 (+/- 0.04746) z^-1 - 0.126 (+/- 0.04527) z^-2 - 0.2089 (+/- 0.04885)
							z^-35 - 0.08375 (+/- 0.0686) z^-36 + 0.1274 (+/- 0.0513) z^-37

C(z) = 1     

Fit to estimation data: 52.68% 
FPE: 0.000849, MSE: 0.0007139

The residual is deemed to be WHITE according to the Monti-test (as  8.57 < 31.41).
The D'Agostino-Pearson's K2 test indicates that the residuals is NOT normal distributed.`
```

We decided to have different naive predictors for different step sizes. The naive predictor for k=1 is just an AR(1) polynomial, based on the assumption that the vegetation should be roughly the same as it was a month ago. For k=7 the naive predictor is an AR(1) convolved with AR(36), using last years value), as it struggles a lot more for the k=7 case.

### Evaluation for validation data

Below are the residuals shown for our model and the naive predictors.  

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 13.43.37.png" alt="Skärmavbild 2024-01-06 kl. 20.20.00" style="zoom:50%;" />

ACF of residuals (left k=1, right k=7)

![Skärmavbild 2024-01-06 kl. 19.16.56](/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-06 kl. 19.16.56.png)

For k=7 the acf should look like an MA(6), which it just barely does. For k=1 it looks white enough. Below we can also see our model and the naive predictor plotted with vegetation.

<img src="/Users/edgar/Desktop/Skärmavbild 2024-01-09 kl. 21.27.08.png" alt="Skärmavbild 2024-01-09 kl. 21.27.08" style="zoom:50%;" />

At k=7 the models just follow the trends and not the smaller spikes as they do for the k=1 case. 

We fitst had an ARMA model with slightly better fit (52.73%) to the model data, but was worse compared to the naive seven step predictor. Probably because it’s harder to find good MA parameters. This made us just choose an AR model instead. Below are the residaul error and variances for the models.  Our model is better than the naive predictors in every statistic.  

```
     models      squared_error    e_normalized_variance    e_variance
    _________    _____________    _____________________    __________
		ARMA k=1       0.066061             0.16121            0.0006735
    ARMA k=7       0.40765              0.58835            0.0041552
    Naive k=1      0.067157             0.18032            0.00067105
    Model k=1      0.060031             0.1552             0.00061256
    Naive k=7      0.40291              0.5826             0.0041081
    Model k=7      0.33593              0.51544            0.0034263

```

### Evaluation for test data

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 21.16.29.png" alt="Skärmavbild 2024-01-06 kl. 20.16.01" style="zoom:50%;" />

```
     models      squared_error    e_normalized_variance    e_variance
    _________    _____________    _____________________    __________

    Naive k=1      0.076754              0.11086            0.0011395
    Model k=1      0.056429             0.097286           0.00082443
    Naive k=7       0.23644              0.35299            0.0028747
    Model k=7       0.25051              0.35039            0.0032031
```

We are actually worse than the naive predictor for k=7. This could be because we made the error of not removing the small trend in the data, because we thought it was to small. The naive predictor has fever parameters so it isn’t as much affected by this error as our larger model is.

## BJ model

Right of the bat we have scaled the input data (x) to fit on the plot. The input data isn’t normal at all either so a box cox transform with  $\lambda =  -0.4864$ is needed. 

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 21.10.40.png" alt="Skärmavbild 2024-01-07 kl. 16.39.35" style="zoom:50%;" />

After the box cox transform the input data still needs to be scaled, but now with factor 4. We want to scale the data to minimize computer precission errors, caused by to large difference in scale of two parameters. 

We won’t do inverse boc cox on the input data, as we only use it to predict the output data (y).

Here we just outline key modeling steps. For a little more detailed explanation of finding BJ model orders prewhitening and so on look at *Theory section B*.

When finding an ARMA model for the input data, we like before had to remove a yearly season (s=36).

```
Discrete-time ARMA model: A(z)y(t) = C(z)e(t)            

  A(z) = 1 - 0.8186 (+/- 0.03044) z^-1 + 0.04849 (+/- 0.01913) z^-8
                 - 0.4172 (+/- 0.04367) z^-36 + 0.2346 (+/- 0.04776) z^-37

  C(z) = 1 + 0.6547 (+/- 0.03774) z^-4 + 0.1464 (+/- 0.03667) z^-10
             
Estimated using PEM. 
Fit to estimation data: 71.63%         
FPE: 0.002433, MSE: 0.002373      


The residual is NOT deemed to be white according to the Monti-test (as 35.02 > 31.41).
The D'Agostino-Pearson's K2 test indicates that the Trans ARMA is NOT normal distributed.
```

Residuals are not white and not normal. We could maybe find a better ARMA model if we had more time and energy, but this is good enough. 

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 21.05.45.png" alt="Skärmavbild 2024-01-09 kl. 21.05.45" style="zoom:50%;" />

Cross corrrelation plot, we identify d=5, r=s=0. We, for some reason, also tried higher r, and s but the model became unstable.

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 21.03.14.png" alt="Skärmavbild 2024-01-07 kl. 16.56.28" style="zoom:50%;" />

```
y(t) =  B(z)u(t) + [C(z)/D(z)]e(t)

B(z) = 0.3009 (+/- 0.0256) z^-5

D(z) = 1 - 0.7485 (+/- 0.04814) z^-1 - 0.1888 (+/- 0.04806) z^-2
          - 0.2901 (+/- 0.04953) z^-36 + 0.01835 (+/- 0.06523) z^-37
                                              + 0.2083 (+/- 0.05142) z^-38


Estimated using PEM
Fit to estimation data: 45.89% (prediction focus)
FPE: 0.001124, MSE: 0.0009335


The residual is deemed to be WHITE according to the Monti-test (as 28 < 31.41).
The D'Agostino-Pearson's K2 test indicates that the  is NOT normal distributed.
```

Residuals aren’t normal here either, and with Monti-test being quite close, we can’t say for certain that the residuals are completely white. But we are still happy, and don’t care if our model could be 5% better by getting perfect white noise.

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 20.59.43.png" alt="Skärmavbild 2024-01-07 kl. 17.05.01" style="zoom:50%;" />

### Validation data

We are using our ARMA model as the naive predictor, it’s not a really “naive” predictor  but we have high hopes for our BJ model, which incidentely does a little bit better.

![Skärmavbild 2024-01-09 kl. 16.32.01](/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 16.32.01.png)

```
     models      squared_error    e_normalized_variance    e_variance
    _________    _____________    _____________________    __________

    Naive k=1      0.067793              0.16476           0.00069107
    Model k=1      0.079854              0.20421           0.00081376
    Naive k=7      0.3895                0.61052           0.0039686
    Model k=7      0.37198               0.5795            0.0036747

```

### Test data

<img src="/Users/edgar/Desktop/Skärmavbild 2024-01-09 kl. 20.44.50.png" alt="Skärmavbild 2024-01-09 kl. 20.44.50" style="zoom:50%;" />

```
     models      squared_error    e_normalized_variance    e_variance
    _________    _____________    _____________________    __________

    Naive k=1      0.056429             0.098421           0.00082443
    Model k=1      0.071737              0.12266            0.0010825
    Naive k=7       0.25051              0.37106            0.0032031
    Model k=7        0.2764              0.35994            0.0037732
```

The BJ model does worse here, this could be because our kalman reconstruction of rain data isn’t as good here, at the end of the dataset, as we have used a backwards and forwards kalman filter. The residuals aren’t MA(6) as we are expecting but, atleast MA(5) which is close enough.

<img src="/Users/edgar/Library/Application Support/typora-user-images/Skärmavbild 2024-01-09 kl. 20.51.12.png" alt="Skärmavbild 2024-01-09 kl. 20.51.12" style="zoom:50%;" />
