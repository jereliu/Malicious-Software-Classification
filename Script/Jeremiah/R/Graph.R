#### Caffè breva
require(magrittr)
require(dplyr)
require(plot3D)
require(xtable)

X = data.frame(
Agent = 114
,
AutoRun = 50
,
FraudLoad =37
,
FraudPack =  32
, 
Hupigon = 41
, 
Krap = 39
, 
Lipler = 53
, 
Magania = 41
, 
None = 1609
, 
Poison = 21
,
Swizzor =  542
,
Tdss =  32
,
VB = 376
,
Virut = 59
,
Zbot = 40
)

rownames(X) = NULL
xtable(X, digit = 0)


#### 1. Readin csv.gz. Data ####
dat <- read.csv("parameters2.csv")

d1 <- unique(dat$max_features)
d2 <- unique(dat$max_depth)

zMat <- matrix(dat$Mis_matches, nrow = length(d1), 
               dimnames = list(unique(dat$max_features), 
                               unique(dat$max_depth)))

xtable(zMat, digits = 6)

persp3D(x = d1, y = d2, z = zMat,
        theta = 30, phi = 30, 
        col = ramp.col(c("red", "orange", "yellow"), n = 100, alpha = 0.8), 
        border = "black",
        xlab = "max_features", ylab = "max_depth",
        ticktype = "detailed")

