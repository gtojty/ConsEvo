require(ggplot2)
require(plyr)
require(GGally)
require(stats)
require(xlsx)
require(lme4)
require(Hmisc)

D <- read.csv("Allres.csv")
D$scen <- factor(D$scen); D$socPres <- factor(D$socPres); D$markedness <- factor(D$markedness); D$comtype <- factor(D$comtype)

# correlations
sub_f <- D[,c("Pop", "minSSD", "numCom")]
rcorr(as.matrix(sub_f), type = c("pearson"))
# Pop minSSD numCom
# Pop     1.00   0.16  -0.51
# minSSD  0.16   1.00  -0.10
# numCom -0.51  -0.10   1.00
# 
# n= 700 


# general linear regression models
lm1 <- glm(minSSD ~ Pop + socPres + markedness + comtype  +
                    #Pop:socPres + Pop:markedness + Pop:comtype + 
                    socPres:comtype, data=D)
summary(lm1)
# Deviance Residuals: 
#   Min          1Q      Median          3Q         Max  
# -0.0023479  -0.0002650  -0.0001453   0.0001943   0.0014817  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)        9.055e-03  5.366e-05 168.755  < 2e-16 ***
#   Pop                6.344e-04  9.964e-05   6.367 3.50e-10 ***
#   socPres1          -1.444e-03  6.302e-05 -22.910  < 2e-16 ***
#   markedness1       -8.622e-05  6.302e-05  -1.368    0.172    
# comtype1           1.014e-04  6.302e-05   1.610    0.108    
# socPres1:comtype1  6.810e-04  8.912e-05   7.641 7.16e-14 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 2.779861e-07)
# 
# Null deviance: 0.00042572  on 699  degrees of freedom
# Residual deviance: 0.00019292  on 694  degrees of freedom
# AIC: -8572.5
# 
# Number of Fisher Scoring iterations: 2

# general linear regression models
lm1a <- glm(minSSD ~ Pop + socPres + markedness + comtype, data=D)
summary(lm1a)
# Deviance Residuals: 
#   Min          1Q      Median          3Q         Max  
# -2.518e-03  -3.593e-04  -7.407e-05   2.544e-04   1.652e-03  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  9.055e-03  5.583e-05 162.191  < 2e-16 ***
#   Pop          6.344e-04  1.037e-04   6.120 1.57e-09 ***
#   socPres1    -1.273e-03  6.133e-05 -20.764  < 2e-16 ***
#   markedness1 -2.565e-04  6.133e-05  -4.182 3.26e-05 ***
#   comtype1     4.419e-04  4.636e-05   9.532  < 2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 3.009407e-07)
# 
# Null deviance: 0.00042572  on 699  degrees of freedom
# Residual deviance: 0.00020915  on 695  degrees of freedom
# AIC: -8518
# 
# Number of Fisher Scoring iterations: 2

lm2 <- glm(numCom ~ Pop + socPres + markedness + comtype +  
             Pop:socPres + Pop:markedness + Pop:comtype, data=D)
summary(lm2)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2265.8   -680.8   -207.5    305.5   8011.1  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)       1588.9      213.1   7.456 2.67e-13 ***
#   Pop              -3453.6      591.1  -5.843 7.89e-09 ***
#   socPres1           876.9      281.9   3.110  0.00194 ** 
#   markedness1        475.6      281.9   1.687  0.09203 .  
# comtype1          -440.5      213.1  -2.067  0.03909 *  
#   Pop:socPres1     -1788.4      781.9  -2.287  0.02248 *  
#   Pop:markedness1  -1000.9      781.9  -1.280  0.20094    
# Pop:comtype1       980.4      591.1   1.659  0.09764 .  
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 1956314)
# 
# Null deviance: 1865479771  on 699  degrees of freedom
# Residual deviance: 1353769518  on 692  degrees of freedom
# AIC: 12137
# 
# Number of Fisher Scoring iterations: 2

lm2a <- glm(numCom ~ Pop + socPres + markedness + comtype, data=D)
summary(lm2a)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1981.4   -693.2   -275.5    253.6   8018.6  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   1806.0      142.7  12.657   <2e-16 ***
#   Pop          -4177.1      265.0 -15.764   <2e-16 ***
#   socPres1       340.4      156.8   2.171   0.0303 *  
#   markedness1    175.4      156.8   1.119   0.2637    
# comtype1      -146.4      118.5  -1.236   0.2170    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 1965884)
# 
# Null deviance: 1865479771  on 699  degrees of freedom
# Residual deviance: 1366289443  on 695  degrees of freedom
# AIC: 12138
# 
# Number of Fisher Scoring iterations: 2

# post-hoc t-test
t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen1a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen1a"]
# t = -0.29536, df = 260.05, p-value = 0.768
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -1.167601e-04  8.630219e-05
# sample estimates:
#   mean of x   mean of y 
# 0.009245314 0.009260543 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen1b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen1b"]
# t = 11.399, df = 236.71, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0005470065 0.0007755851
# sample estimates:
#   mean of x   mean of y 
# 0.009245314 0.008584019 

t.test(D$minSSD[D$scen=='Scen1a'], D$minSSD[D$scen=='Scen1b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1a"] and D$minSSD[D$scen == "Scen1b"]
# t = 10.612, df = 269.83, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0005510125 0.0008020370
# sample estimates:
#   mean of x   mean of y 
# 0.009260543 0.008584019 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen2a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen2a"]
# t = 2.0399, df = 275.41, p-value = 0.04232
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   3.011960e-06 1.694366e-04
# sample estimates:
#   mean of x   mean of y 
# 0.009245314 0.009159090 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen2b"]
# t = 19.255, df = 195.19, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.001295870 0.001591622
# sample estimates:
#   mean of x   mean of y 
# 0.009245314 0.007801568 

t.test(D$minSSD[D$scen=='Scen2a'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen2a"] and D$minSSD[D$scen == "Scen2b"]
# t = 18.391, df = 185.89, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.001211897 0.001503146
# sample estimates:
#   mean of x   mean of y 
# 0.009159090 0.007801568 

t.test(D$minSSD[D$scen=='Scen1a'], D$minSSD[D$scen=='Scen2a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1a"] and D$minSSD[D$scen == "Scen2a"]
# t = 2.0351, df = 247.54, p-value = 0.04291
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   3.264126e-06 1.996424e-04
# sample estimates:
#   mean of x   mean of y 
# 0.009260543 0.009159090 

t.test(D$minSSD[D$scen=='Scen1b'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1b"] and D$minSSD[D$scen == "Scen2b"]
# t = 9.3336, df = 252.02, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0006173513 0.0009475492
# sample estimates:
#   mean of x   mean of y 
# 0.008584019 0.007801568 
