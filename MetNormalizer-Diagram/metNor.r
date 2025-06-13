metNor <- function (ms1.data.name = "data.csv", sample.info.name = "sample.info.csv", 
    minfrac.qc = 0, minfrac.sample = 0, optimization = TRUE, 
    multiple = 5, threads = 3, path = ".") 
{
    options(warn = -1)
    cat(crayon::green("Checking data...\n"))
    check_result <- checkData(data = ms1.data.name, sample.info = sample.info.name, 
        path = path)
    if (any(as.numeric(check_result[, "Error"]) > 0)) {
        stop("Error in your data or sample information.\n")
    }
    dir.create(path, showWarnings = FALSE)
    cat(crayon::green("Reading data...\n"))
    data <- readr::read_csv(file.path(path, "data.csv"), col_types = readr::cols(), 
        progress = FALSE) %>% as.data.frame()
    sample.info <- readr::read_csv(file.path(path, "sample.info.csv"), 
        col_types = readr::cols()) %>% dplyr::arrange(injection.order)
    sample.order <- sample.info %>% dplyr::filter(class == "Subject") %>% 
        dplyr::pull(injection.order) %>% as.numeric()
    qc.order <- sample.info %>% dplyr::filter(class == "QC") %>% 
        dplyr::pull(injection.order) %>% as.numeric()
    tags <- data %>% dplyr::select(-dplyr::one_of(sample.info$sample.name))
    sample.name <- sample.info %>% dplyr::filter(class == "Subject") %>% 
        dplyr::pull(sample.name)
    qc.name <- sample.info %>% dplyr::filter(class == "QC") %>% 
        dplyr::pull(sample.name)
    sample <- data %>% dplyr::select(dplyr::one_of(sample.name))
    qc <- data %>% dplyr::select(dplyr::one_of(qc.name))
    rownames(sample) <- rownames(qc) <- tags$name
    cat(crayon::green("Filtering data...\n"))
    qc.per <- apply(qc, 1, function(x) {
        sum(x != 0)/ncol(qc)
    })
    sample.per <- apply(sample, 1, function(x) {
        sum(x != 0)/ncol(sample)
    })
    remain.idx <- which(qc.per >= minfrac.qc & sample.per >= 
        minfrac.sample)
    if (length(remain.idx) > 0) {
        sample <- sample[remain.idx, , drop = FALSE]
        qc <- qc[remain.idx, , drop = FALSE]
        tags <- tags[remain.idx, , drop = FALSE]
    }
    sample <- t(sample)
    qc <- t(qc)
    tags <- t(tags)
    cat(crayon::red("OK\n"))
    cat(crayon::green("SVR normalization...\n"))
    SXTsvrNor(sample = sample, QC = qc, tags = tags, sample.order = sample.order, 
        QC.order = qc.order, multiple = multiple, path = path, 
        rerun = TRUE, peakplot = FALSE, datastyle = "tof", dimension1 = TRUE, 
        threads = threads)
    cat(crayon::bgRed("All done!\n"))
}