suppressMessages({library(MASS); library(pscl)})
set.seed(7)
out <- list()
for (i in 1:12) {
  n <- sample(c(30, 80, 200), 1); mu <- runif(1, 1, 200); th <- runif(1, 0.3, 8); pz <- sample(c(0, 0.1, 0.3), 1)
  y <- rnbinom(n, mu = mu, size = th); y[runif(n) < pz] <- 0
  sf <- runif(n, 0.5, 1.5); o <- log(sf)
  f1 <- glm.nb(y ~ 1); f2 <- glm.nb(y ~ offset(o))
  z1 <- tryCatch(zeroinfl(y ~ 1 | 1, dist = "negbin"), error=function(e) NULL)
  z2 <- tryCatch(zeroinfl(y ~ offset(o) | 1, dist = "negbin"), error=function(e) NULL)
  zr <- function(z) if (is.null(z)) c(NA,NA,NA,NA) else c(exp(z$coef$count[1]), z$theta, plogis(z$coef$zero[1]), as.numeric(logLik(z)))
  write.csv(data.frame(y=y, o=o), sprintf("d%02d.csv", i), row.names=FALSE)
  out[[i]] <- c(i, exp(coef(f1)), f1$theta, as.numeric(logLik(f1)), exp(coef(f2)), f2$theta, as.numeric(logLik(f2)), zr(z1), zr(z2))
}
m <- do.call(rbind, out)
colnames(m) <- c("i","nb_mu","nb_th","nb_ll","nbo_mu","nbo_th","nbo_ll","z_mu","z_th","z_pi","z_ll","zo_mu","zo_th","zo_pi","zo_ll")
write.csv(m, "r_fits.csv", row.names=FALSE)
