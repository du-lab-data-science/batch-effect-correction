svr_function <- function (index, sample, QC, sample.order, QC.order, multiple) 
{
    colnames(sample) <- colnames(QC)
    sample <- sample[, index, drop = FALSE]
    QC <- QC[, index, drop = FALSE]
    data.order <- c(sample.order, QC.order)
    data.nor <- lapply(c(1:ncol(sample)), function(i) {
        if (multiple != 1) {
            correlation <- abs(cor(x = rbind(sample, QC)[, i], 
                y = rbind(sample, QC))[1, ])
            cor.peak <- match(names(sort(correlation, decreasing = TRUE)[1:6][-1]), 
                names(correlation))
            rm(list = "correlation")
            svr.reg <- e1071::svm(QC[, cor.peak], QC[, i])
        }
        else {
            svr.reg <- e1071::svm(unlist(QC[, i]) ~ QC.order)
        }
        predict.QC <- summary(svr.reg)$fitted
        QC.nor1 <- QC[, i]/predict.QC
        QC.nor1[is.nan(unlist(QC.nor1))] <- 0
        QC.nor1[is.infinite(unlist(QC.nor1))] <- 0
        QC.nor1[is.na(unlist(QC.nor1))] <- 0
        QC.nor1[which(unlist(QC.nor1) < 0)] <- 0
        if (multiple != 1) {
            predict.sample <- predict(svr.reg, sample[, cor.peak])
        }
        else {
            predict.sample <- predict(svr.reg, data.frame(QC.order = c(sample.order)))
        }
        sample.nor1 <- sample[, i]/predict.sample
        sample.nor1[is.nan(unlist(sample.nor1))] <- 0
        sample.nor1[is.infinite(unlist(sample.nor1))] <- 0
        sample.nor1[is.na(unlist(sample.nor1))] <- 0
        sample.nor1[which(unlist(sample.nor1) < 0)] <- 0
        return(list(sample.nor1, QC.nor1))
    })
    sample.nor <- lapply(data.nor, function(x) x[[1]])
    QC.nor <- lapply(data.nor, function(x) x[[2]])
    rm(list = "data.nor")
    sample.nor <- t(do.call(rbind, sample.nor))
    QC.nor <- t(do.call(rbind, QC.nor))
    colnames(sample.nor) <- colnames(QC.nor) <- colnames(sample)
    rm(list = c("sample", "QC"))
    svr.data <- list(sample.nor = sample.nor, QC.nor = QC.nor, 
        index = index)
    rm(list = c("sample.nor", "QC.nor"))
    return(svr.data)
}