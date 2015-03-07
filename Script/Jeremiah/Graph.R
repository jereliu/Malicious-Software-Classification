library(xtable)

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
xtable(X)
