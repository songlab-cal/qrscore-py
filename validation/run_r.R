suppressMessages({library(dplyr); library(MASS); library(pscl); library(arrangements); library(BiocParallel)})
simplex.sample <- function(n, N) { e <- matrix(rexp(n*N), N); list(samples = e / rowSums(e)) }  # hitandrun shim
for (f in c("utils.R","QRscore.R","de_analysis.R")) source(file.path(Sys.getenv("QRSCORE_R_SRC", "QRscore/R"), f)); stopifnot(exists(".BETA_CAP"))
# Deterministic tie-breaking so R and Python see identical ranks (y placed after tied x).
rank_x <- function(x, y = NULL, ties.break = TRUE) {
  s <- sort(c(x, y)); d <- diff(s); g <- if (any(d > 0)) min(d[d > 0]) else 1
  y <- y + g / 4; xo <- sort(x); vapply(xo, function(v) sum(c(x, y) < v) + 1, numeric(1)) }
res <- list(); add <- function(case, value) res[[length(res) + 1]] <<- data.frame(case = case, r = as.numeric(value))
q <- function(e) suppressWarnings(suppressMessages(e))

# 1. weights
for (pr in list(c(2,10,.1,50,25), c(0.5,200,0,300,120), c(5,40,.1,100,60), c(100,3,.3,20,9), c(1.3,1500,.05,80,40), c(1e4,5,.1,60,40), c(1e6,1000,0,60,40), c(1e6,3,.2,60,40), c(100,9000,0,60,40))) {
  wm <- computeweight_mean(pr[1], pr[2], pr[3], pr[4], pr[5]); wd <- computeweight_disp(pr[1], pr[2], pr[3], pr[4], pr[5])
  tag <- paste(pr, collapse="_")
  add(paste0("wmean_w_", tag, "_", seq_along(wm$weight)), wm$weight); add(paste0("wmean_v_", tag), wm$var)
  add(paste0("wdisp_w_", tag, "_", seq_along(wd$weight)), wd$weight); add(paste0("wdisp_v_", tag), wd$var) }
# 2. exact composition p-values and asymptotic moments
for (alt in c("two.sided","greater","less")) {
  add(paste0("comp_exact_", alt), q(getCompositionPValue(0.31, 25, 6, 1, (5:0)/5, alt, "unbiased", 10)))
  add(paste0("comp_exact2_", alt), q(getCompositionPValue(0.2, 30, 5, 2, c(1,.2,0,.2,1), alt, "unbiased", 10))) }
set.seed(3); xa <- abs(rnorm(120)); ya <- abs(rnorm(150, sd = 1.3))
write.csv(data.frame(v = xa), "flex_x.csv", row.names = FALSE); write.csv(data.frame(v = ya), "flex_y.csv", row.names = FALSE)
for (p in 1:2) for (alt in c("two.sided","greater","less")) add(paste0("flex_asym_p", p, "_", alt),
  q(QRscore_Flex(xa[1:60], ya, p, c(60:0, 1:0)[1:61]^p, alt, "asymptotic")))
# 3. Mann-Whitney through QRscoreTest
set.seed(4); m1 <- rnorm(20); m2 <- rnorm(25, 1); m3 <- round(rnorm(70) * 3); m4 <- round(rnorm(80) * 3 + 1)
write.csv(data.frame(v = c(m1, m2, m3, m4)), "mw.csv", row.names = FALSE)
for (alt in c("two.sided","greater","less")) {
  add(paste0("mw_exact_", alt), q(QRscoreTest(c(m1, m2), rep(c("A","B"), c(20, 25)), p = 1, wList = 20:0, alternative = alt)))
  add(paste0("mw_ties_", alt), q(QRscoreTest(c(m3, m4), rep(c("A","B"), c(70, 80)), p = 1, wList = 70:0, alternative = alt))) }
# 4. QRscore_ZINB / QRscoreTest on simulated counts
set.seed(11); cases <- list()
for (i in 1:16) {
  nx <- sample(c(8, 30, 60, 120), 1); ny <- sample(c(9, 40, 110, 150), 1); pz <- sample(c(0, 0.15), 1)
  x <- rzinbinom(nx, size = runif(1, .5, 4), mu = runif(1, 5, 80), pi = pz); y <- rzinbinom(ny, size = runif(1, .5, 4), mu = runif(1, 5, 80), pi = pz)
  sf <- runif(nx + ny, .5, 1.5)
  write.csv(data.frame(v = c(x, y), g = rep(c("A", "B"), c(nx, ny)), sf = sf), sprintf("z%02d.csv", i), row.names = FALSE)
  for (meas in c("mean", "dispersion")) for (zi in c(TRUE, FALSE)) for (lr in c(FALSE, TRUE)) {
    tag <- sprintf("z%02d_%s_zi%d_lr%d", i, meas, zi, lr)
    add(paste0(tag, "_asym"), q(QRscore_ZINB(x, y, zero_inflation = zi, LR.test = lr, approx = "asymptotic", measure = meas)))
    add(paste0(tag, "_asym_sf"), q(QRscore_ZINB(x, y, size_factors = sf, zero_inflation = zi, LR.test = lr, approx = "asymptotic", measure = meas)))
    if (nx < 10 && ny < 10) add(paste0(tag, "_exact"), q(QRscore_ZINB(x, y, zero_inflation = zi, LR.test = lr, approx = "resample", measure = meas)))
  } }
# 5. K-sample
set.seed(12)
for (i in 1:8) {
  ns <- sample(c(40, 100, 150), 3, TRUE); pz <- sample(c(0, .1), 1)
  v <- unlist(lapply(1:3, function(j) rzinbinom(ns[j], size = runif(1, .7, 4), mu = runif(1, 10, 50), pi = pz)))
  lab <- rep(c("G2", "G1", "G3"), ns); sf <- runif(sum(ns), .5, 1.5)
  write.csv(data.frame(v = v, g = lab, sf = sf), sprintf("k%02d.csv", i), row.names = FALSE)
  for (meas in c("mean", "dispersion")) for (zi in c(TRUE, FALSE)) {
    tag <- sprintf("k%02d_%s_zi%d", i, meas, zi)
    add(tag, q(QRscoreTest(v, lab, approx = "asymptotic", zero_inflation = zi, measure = meas)))
    add(paste0(tag, "_sf"), q(QRscoreTest(v, lab, size_factors = sf, approx = "asymptotic", zero_inflation = zi, measure = meas))) } }

# 7. New code paths: glm.nb divergence (d01..d12), size-factor swap
for (i in 1:12) { d <- read.csv(sprintf("d%02d.csv", i)); y <- d$y; h <- length(y) %/% 2
  for (meas in c("mean","dispersion")) for (lr in c(FALSE, TRUE)) {
    tag <- sprintf("nbdiv%02d_%s_lr%d", i, meas, lr)
    add(tag, q(QRscore_ZINB(y[1:h], y[(h+1):length(y)], zero_inflation = lr, LR.test = lr, approx = "asymptotic", measure = meas)))
    add(paste0(tag, "_sf"), q(QRscore_ZINB(y[1:h], y[(h+1):length(y)], size_factors = exp(d$o), zero_inflation = lr, LR.test = lr, approx = "asymptotic", measure = meas))) } }
for (i in 1:6) { d <- read.csv(sprintf("z%02d.csv", i + 6)); big <- rep(c("A","B"), c(sum(d$g=="B"), sum(d$g=="A")))   # relabel so first group is larger when B was larger
  for (meas in c("mean","dispersion")) add(sprintf("swap%02d_%s", i, meas), q(QRscoreTest(d$v, ifelse(d$g == "A", "Z", "A"), size_factors = d$sf, measure = meas, zero_inflation = FALSE, approx = "asymptotic"))) }
out <- do.call(rbind, res); write.csv(out, "r_results.csv", row.names = FALSE); cat(nrow(out), "results (with new cases)\n")


# 6. QRscoreGenetest on the bundled data (first 150 genes)
load(file.path(Sys.getenv("QRSCORE_R_SRC", "QRscore/R"), "../data/example_dataset.rda")); e <- example_dataset
g2 <- q(QRscoreGenetest(e$example_data2[1:150, ], e$labels2, e$size_factors2, pairwise_test = TRUE, pairwise_logFC = TRUE, test_mean = TRUE, test_dispersion = TRUE, approx = "asymptotic", rank = FALSE))
write.csv(g2$mean_test, "gt2_mean.csv"); write.csv(g2$var_test, "gt2_var.csv")
g1 <- q(QRscoreGenetest(e$example_data1[1:150, ], e$labels1, e$size_factors1, pairwise_test = FALSE, pairwise_logFC = TRUE, test_mean = TRUE, test_dispersion = TRUE, approx = "asymptotic", rank = FALSE))
write.csv(g1$mean_test, "gt1_mean.csv"); write.csv(g1$var_test, "gt1_var.csv")
