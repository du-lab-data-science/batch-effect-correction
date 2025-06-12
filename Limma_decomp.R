y <- matrix(rnorm(10*9),10,9)
y[,1:3] <- y[,1:3] + 5
batch <- c("A","A","A","B","B","B","C","C","C")

RBEC <- function (x, batch = NULL, batch2 = NULL, covariates = NULL, 
          design = NULL, group = NULL, ...) 
{
  browser()
  if (is.null(batch) && is.null(batch2) && is.null(covariates)) 
    return(as.matrix(x))
  if (!is.null(batch)) {
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop = FALSE]
  }
  if (!is.null(batch2)) {
    batch2 <- as.factor(batch2)
    contrasts(batch2) <- contr.sum(levels(batch2))
    batch2 <- model.matrix(~batch2)[, -1, drop = FALSE]
  }
  if (!is.null(covariates)) {
    covariates <- as.matrix(covariates)
    covariates <- t(t(covariates) - colMeans(covariates))
  }
  X.batch <- cbind(batch, batch2, covariates)
  if (!is.null(group)) {
    group <- as.factor(group)
    design <- model.matrix(~group)
  }
  if (is.null(design)) {
    message("design matrix of interest not specified. Assuming a one-group experiment.")
    design <- matrix(1, ncol(x), 1)
  }
  x <- as.matrix(x)
  fit <- lmFit(x, cbind(design, X.batch), ...)
  beta <- fit$coefficients[, -(1:ncol(design)), drop = FALSE]
  beta[is.na(beta)] <- 0
  x - beta %*% t(X.batch)
}