suppressMessages({library(dplyr); library(MASS); library(pscl); library(arrangements)})
simplex.sample <- function(n, N) { e <- matrix(rexp(n*N), N); list(samples = e / rowSums(e)) }
for (f in c("utils.R","QRscore.R")) source(file.path(Sys.getenv("QRSCORE_R_SRC", "QRscore/R"), f))
rank_x <- function(x, y = NULL, ties.break = TRUE) { s <- sort(c(x, y)); d <- diff(s); g <- if (any(d > 0)) min(d[d > 0]) else 1
  y <- y + g / 4; xo <- sort(x); vapply(xo, function(v) sum(c(x, y) < v) + 1, numeric(1)) }
set.seed(5); x1 <- abs(rnorm(12)); x2 <- abs(rnorm(30)); y2 <- abs(rnorm(90, sd = 1.4))
z <- read.csv("z03.csv"); zx <- z$v[z$g == "A"]; zy <- z$v[z$g == "B"]
write.csv(data.frame(v = c(x1, x2, y2)), "mc_in.csv", row.names = FALSE)
q <- function(e) suppressWarnings(suppressMessages(e)); N <- 200000
r <- c(one_p2 = q(QRscore_Flex(x1, NULL, 2, rep(1, 12), "two.sided", "resample", "unbiased", NULL, N)),
       one_p1g = q(QRscore_Flex(x1, NULL, 1, 12:1, "greater", "resample", "valid", NULL, N)),
       two_p2 = q(QRscore_Flex(x2, y2, 2, rep(1, 31), "two.sided", "resample", "unbiased", NULL, N)),
       two_p1l = q(QRscore_Flex(x2, y2, 1, c(15:0, 1:15), "less", "resample", "unbiased", NULL, N)),
       zinb_rs_mean = q(QRscore_ZINB(zx, zy, approx = "resample", resamp_num = N, measure = "mean", zero_inflation = FALSE)),
       zinb_rs_disp = q(QRscore_ZINB(zx, zy, approx = "resample", resamp_num = N, measure = "dispersion", zero_inflation = FALSE)))
cat(length(zx), length(zy), "\n"); write.csv(data.frame(case = names(r), r = r), "mc_r.csv", row.names = FALSE)
