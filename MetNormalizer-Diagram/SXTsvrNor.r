SXTsvrNor <- function (sample, QC, tags, sample.order, QC.order, multiple = 5, 
    rerun = TRUE, peakplot = TRUE, path = ".", datastyle = "tof", 
    dimension1 = TRUE, threads = 1) 
{
    options(warn = -1)
    if (is.null(path)) {
        path <- getwd()
    }
    else {
        dir.create(path, showWarnings = FALSE)
    }
    output_path <- file.path(path, "svr_normalization_result")
    dir.create(output_path, showWarnings = FALSE)
    if (!rerun) {
        cat("Use previous normalization data\n")
        load(file.path(output_path, "normalization_file"))
    }
    else {
        ichunks <- split((1:ncol(sample)), 1:threads)
        svr.data <- BiocParallel::bplapply(ichunks, FUN = svr_function, 
            BPPARAM = BiocParallel::MulticoreParam(workers = threads, 
                progressbar = TRUE), sample = sample, QC = QC, 
            sample.order = sample.order, QC.order = QC.order, 
            multiple = multiple)
        sample.nor <- lapply(svr.data, function(x) {
            x[[1]]
        })
        QC.nor <- lapply(svr.data, function(x) {
            x[[2]]
        })
        index <- lapply(svr.data, function(x) {
            x[[3]]
        })
        sample.nor <- do.call(cbind, sample.nor)
        QC.nor <- do.call(cbind, QC.nor)
        index <- unlist(index)
        sample.nor <- sample.nor[, order(index)]
        QC.nor <- QC.nor[, order(index)]
        QC.median <- apply(QC, 2, median)
        if (dimension1) {
            QC.nor <- t(t(QC.nor) * QC.median)
            sample.nor <- t(t(sample.nor) * QC.median)
        }
        save(QC.nor, sample.nor, file = file.path(output_path, 
            "normalization_file"))
    }
    rsd <- function(x) {
        x <- sd(x) * 100/mean(x)
    }
    sample.rsd <- apply(sample, 2, rsd)
    sample.nor.rsd <- apply(sample.nor, 2, rsd)
    QC.rsd <- apply(QC, 2, rsd)
    QC.nor.rsd <- apply(QC.nor, 2, rsd)
    sample.no.nor <- rbind(tags, sample.rsd, QC.rsd, sample, 
        QC)
    sample.svr <- rbind(tags, sample.nor.rsd, QC.nor.rsd, sample.nor, 
        QC.nor)
    save(sample.nor, QC.nor, tags, sample.order, QC.order, file = file.path(output_path, 
        "data_svr_normalization"))
    write.csv(t(sample.svr), file.path(output_path, "data_svr_normalization.csv"))
    if (peakplot) {
        cat(crayon::green("Drawing peak plots...\n"))
        peak_plot_path <- file.path(output_path, "peak_plot")
        dir.create(peak_plot_path)
        options(warn = -1)
        ichunks <- split((1:ncol(sample)), 1:threads)
        BiocParallel::bplapply(ichunks, FUN = peak_plot, BPPARAM = BiocParallel::MulticoreParam(workers = threads, 
            progressbar = TRUE), sample = sample, sample.nor = sample.nor, 
            QC = QC, QC.nor = QC.nor, sample.order = sample.order, 
            QC.order = QC.order, tags = tags, path = peak_plot_path, 
            sample.rsd = sample.rsd, QC.rsd = QC.rsd, sample.nor.rsd = sample.nor.rsd, 
            QC.nor.rsd = QC.nor.rsd)
    }
    compare_rsd(sample.rsd = sample.rsd, sample.nor.rsd = sample.nor.rsd, 
        QC.rsd = QC.rsd, QC.nor.rsd = QC.nor.rsd, path = output_path)
    options(warn = 0)
    cat("SVR normalization is done\n")
}
