#!/usr/bin/env Rscript
# exp_locfit_probe_v2.R -- v1 at the program's evidentiary standard: reps=15
# (matching the battery convention), mean +/- sd and range printed so the
# bias-domination reading on sharp targets is verifiable, not inferred.
# Run:  Rscript exp_locfit_probe_v2.R
library(locfit)
xg <- seq(-8, 8, length.out=512)
mw <- list(
  Gaussian = list(w=c(1), m=c(0), s=c(1)),
  StronglySkewed = list(w=rep(1/8,8), m=3*((2/3)^(0:7)-1), s=(2/3)^(0:7)),
  Claw = list(w=c(0.5, rep(0.1,5)), m=c(0, (0:4)/2-1), s=c(1, rep(0.1,5)))
)
pdfmix <- function(p, x) {
  out <- 0*x
  for (i in seq_along(p$w)) out <- out + p$w[i]*dnorm(x, p$m[i], p$s[i])
  out
}
sampmix <- function(p, n) {
  idx <- sample(seq_along(p$w), n, replace=TRUE, prob=p$w)
  rnorm(n, p$m[idx], p$s[idx])
}
cat(sprintf("%-16s %6s %12s %8s %20s\n", "target", "n", "mean ISEx1e3", "sd", "range"))
for (tname in names(mw)) {
  p <- mw[[tname]]; ftrue <- pdfmix(p, xg)
  for (n in c(1000, 2000)) {
    ises <- c()
    for (r in 1:15) {
      set.seed(20260627 + 7919*r + n)
      d <- sampmix(p, n)
      fit <- locfit(~lp(d, nn=0.15, deg=2), family="density")
      fh <- predict(fit, newdata=xg)
      fh <- pmax(fh, 0); fh <- fh/sum(fh*(xg[2]-xg[1]))
      ises <- c(ises, 1e3*sum((fh-ftrue)^2)*(xg[2]-xg[1]))
    }
    cat(sprintf("%-16s %6d %12.4f %8.4f %9.4f--%-9.4f\n",
        tname, n, mean(ises), sd(ises), min(ises), max(ises)))
  }
}
cat("tool: exp_locfit_probe_v2.R ; reps=15 ; seed lineage 20260627\n")
