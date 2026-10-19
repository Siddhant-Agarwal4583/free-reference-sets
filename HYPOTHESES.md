# Pre-registered hypotheses (written 2026-10-18, before the main experiment runs)

H1 (Exchangeability gap). OOB confidence scores come from a ~e^-1 sub-ensemble and are
systematically lower than full-ensemble confidence on fresh points. Naive OOB-DoC / OOB-ATC
are therefore biased (underestimate accuracy) in-distribution; the bias shrinks as T grows;
size-matched (Bernoulli-masked) target scoring removes it.

H2 (Free reference set). For each estimator family X in {plain, DoC, ATC, IW}, the OOB variant
has lower absolute estimation error than the holdout variant using a 20% labeled split, with
the largest advantage at small source sample size n.

H3 (No accuracy tax). The model trained on all source data (OOB route) is more accurate on the
target than the model trained on 80% (holdout route).

H4 (Shift-type dependence). Importance weighting is best under subpopulation covariate shift
(shared support); confidence-based methods (DoC/ATC) are best under corruption (support) shift;
all methods degrade under label shift.

H5 (Generality). Conclusions transfer from Random Forests to bootstrapped Extra-Trees.

Outcomes will be reported whether they confirm or refute these hypotheses.
