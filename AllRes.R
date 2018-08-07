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
# rcorr(as.matrix(sub_f), type = c("pearson"))
# Pop minSSD numCom
# Pop     1.00   0.62  -0.48
# minSSD  0.62   1.00  -0.24
# numCom -0.48  -0.24   1.00
# 
# n= 800 

# general linear regression models
lm1 <- glm(minSSD ~ Pop + socPres + markedness + comtype  +
                    #Pop:socPres + Pop:markedness + Pop:comtype + 
                    socPres:comtype, data=D)
summary(lm1)
# Deviance Residuals: 
#   Min          1Q      Median          3Q         Max  
# -2.414e-03  -4.467e-04  -6.505e-05   2.756e-04   1.787e-03  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)        8.732e-03  5.786e-05 150.909  < 2e-16 ***
#   Pop                1.832e-03  6.843e-05  26.766  < 2e-16 ***
#   socPres1          -1.054e-03  6.928e-05 -15.215  < 2e-16 ***
#   markedness1       -6.838e-05  6.928e-05  -0.987    0.324    
# comtype1           8.624e-05  6.928e-05   1.245    0.214    
# socPres1:comtype1  4.408e-04  9.798e-05   4.499 7.83e-06 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 3.839658e-07)
# 
# Null deviance: 0.00071778  on 799  degrees of freedom
# Residual deviance: 0.00030487  on 794  degrees of freedom
# AIC: -9539.9
# 
# Number of Fisher Scoring iterations: 2

# general linear regression models
lm1a <- glm(minSSD ~ Pop + socPres + markedness + comtype, data=D)
summary(lm1a)
# Deviance Residuals: 
#   Min          1Q      Median          3Q         Max  
# -2.525e-03  -3.904e-04  -4.891e-05   3.007e-04   1.897e-03  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  8.732e-03  5.856e-05 149.115  < 2e-16 ***
#   Pop          1.832e-03  6.925e-05  26.448  < 2e-16 ***
#   socPres1    -9.439e-04  6.558e-05 -14.392  < 2e-16 ***
#   markedness1 -1.786e-04  6.558e-05  -2.723  0.00661 ** 
#   comtype1     3.067e-04  4.958e-05   6.185  9.9e-10 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 3.932604e-07)
# 
# Null deviance: 0.00071778  on 799  degrees of freedom
# Residual deviance: 0.00031264  on 795  degrees of freedom
# AIC: -9521.7
# 
# Number of Fisher Scoring iterations: 2

lm2 <- glm(numCom ~ Pop + socPres + markedness + comtype +  
             Pop:socPres + Pop:markedness + Pop:comtype, data=D)
summary(lm2)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1855.9   -744.3   -200.7    221.9   8276.6  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)       1333.0      185.5   7.188 1.52e-12 ***
#   Pop              -1887.2      335.8  -5.620 2.65e-08 ***
#   socPres1           722.9      245.3   2.947  0.00331 ** 
#   markedness1        390.4      245.3   1.591  0.11193    
# comtype1          -354.8      185.5  -1.913  0.05606 .  
# Pop:socPres1     -1043.9      444.2  -2.350  0.01902 *  
#   Pop:markedness1   -563.4      444.2  -1.268  0.20506    
# Pop:comtype1       527.4      335.8   1.571  0.11666    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 1849350)
# 
# Null deviance: 1924637950  on 799  degrees of freedom
# Residual deviance: 1464684921  on 792  degrees of freedom
# AIC: 13825
# 
# Number of Fisher Scoring iterations: 2

lm2a <- glm(numCom ~ Pop + socPres + markedness + comtype, data=D)
summary(lm2a)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1664.2   -748.6   -157.1    192.3   8335.8  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   1527.4      127.3  12.001   <2e-16 ***
#   Pop          -2319.1      150.5 -15.408   <2e-16 ***
#   socPres1       253.1      142.5   1.776   0.0762 .  
# markedness1    136.9      142.5   0.960   0.3372    
# comtype1      -117.5      107.8  -1.090   0.2758    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for gaussian family taken to be 1857684)
# 
# Null deviance: 1924637950  on 799  degrees of freedom
# Residual deviance: 1476858440  on 795  degrees of freedom
# AIC: 13825
# 
# Number of Fisher Scoring iterations: 2

# post-hoc t-test
t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen1a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen1a"]
# t = -0.2815, df = 314.06, p-value = 0.7785
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -0.0001427187  0.0001069920
# sample estimates:
#   mean of x   mean of y 
# 0.009556106 0.009573970 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen1b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen1b"]
# t = 1.1238, df = 317.63, p-value = 0.2619
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -5.132917e-05  1.880890e-04
# sample estimates:
#   mean of x   mean of y 
# 0.009556106 0.009487726

t.test(D$minSSD[D$scen=='Scen1a'], D$minSSD[D$scen=='Scen1b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1a"] and D$minSSD[D$scen == "Scen1b"]
# t = 1.3383, df = 316.07, p-value = 0.1818
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -4.055046e-05  2.130370e-04
# sample estimates:
#   mean of x   mean of y 
# 0.009573970 0.009487726 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen2a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen2a"]
# t = 6.2401, df = 254.66, p-value = 1.814e-09
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0003607123 0.0006933733
# sample estimates:
#   mean of x   mean of y 
# 0.009556106 0.009029063 

t.test(D$minSSD[D$scen=='Scen0'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen0"] and D$minSSD[D$scen == "Scen2b"]
# t = 9.141, df = 207.23, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0008267665 0.0012814579
# sample estimates:
#   mean of x   mean of y 
# 0.009556106 0.008501994 

t.test(D$minSSD[D$scen=='Scen2a'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen2a"] and D$minSSD[D$scen == "Scen2b"]
# t = 4.0597, df = 280.49, p-value = 6.379e-05
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0002715041 0.0007826346
# sample estimates:
#   mean of x   mean of y 
# 0.009029063 0.008501994

t.test(D$minSSD[D$scen=='Scen1a'], D$minSSD[D$scen=='Scen2a'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1a"] and D$minSSD[D$scen == "Scen2a"]
# t = 6.2568, df = 272.31, p-value = 1.518e-09
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0003734514 0.0007163609
# sample estimates:
#   mean of x   mean of y 
# 0.009573970 0.009029063

t.test(D$minSSD[D$scen=='Scen1b'], D$minSSD[D$scen=='Scen2b'], alternative = "two.sided")
# Welch Two Sample t-test
# 
# data:  D$minSSD[D$scen == "Scen1b"] and D$minSSD[D$scen == "Scen2b"]
# t = 8.5077, df = 210.46, p-value = 3.322e-15
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   0.0007573309 0.0012141336
# sample estimates:
#   mean of x   mean of y 
# 0.009487726 0.008501994
