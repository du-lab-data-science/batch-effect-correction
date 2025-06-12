setwd("/Users/jaileru/GitHub/batch-effect-correction-algorithms/")
library(limma)
library(dplyr)
library(readr)
library(Biobase)

#Read in Data 
D <- readr::read_csv("Data/2-area_data_detection_filter.csv")
M <- readr::read_csv("Data/sample_metadata_all_batches_noblanks_nooutliers.csv")

#Set Row Names
D <- D %>% column_to_rownames("...1")
M <- M %>% column_to_rownames("sample_name")

#Ensure row names in metadata and data match 
M <- M[match(rownames(D),rownames(M)), ]

#Get Batch Labels
batch <- M$batch

#Construct Expression Matrix dtype for limma 
mat <- t(as.matrix(D))

sample_data <- Biobase::AnnotatedDataFrame(data = M)
eset <- ExpressionSet(assayData = mat,
                      phenoData = sample_data)
#Specify design matrix (experimental factors that you to preserve)
design <- model.matrix(~sample_type)

#Remove Batch Effects 
corrected <- limma::removeBatchEffect(eset,batch=batch,design=design)
