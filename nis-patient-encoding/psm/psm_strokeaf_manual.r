# Load matching library.
# install.packages(c("Matching"))
if(!require(Matching)) install.packages("Matching",repos = "http://cran.us.r-project.org")
if(!require(foreach)) install.packages("foreach",repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel",repos = "http://cran.us.r-project.org")
library(Matching)
library(foreach)
library(doParallel)

# Data loading
fn = "/home/aisinai/work/repos/nis_patient_encoding/study_type/stroke_af_1/psm_result/FilteredDataForPSM.csv"
data = read.csv(fn, stringsAsFactors=FALSE)

data$CASECONTROL[data$CASECONTROL == 'case'] <- as.logical(TRUE)
data$CASECONTROL[data$CASECONTROL == 'control'] <- as.logical(FALSE)

data$CASECONTROL <- as.logical(data$CASECONTROL)

# Separate out
treatment <- data$CASECONTROL
# treatment = data$CHRONB01

# Identify all features that we can match on.
# age, obesity, liver disease, renal failure, coagulopathy, and the components of the CHA2DS2VASc score
all_features = c(
        'AGE', 'FEMALE', 'CM_OBESE', 
        'Chronic Kidney Disease',
        'Chronic Coagulopathy',
        'CHADS_VASC')

all_features = make.names(all_features)

features_to_use = all_features
match_features = data[, features_to_use]

print(features_to_use)

match_features[] <- lapply(match_features, as.numeric)

rr = Match(Tr=treatment, X=match_features, M=1, ties=FALSE, replace=FALSE)

write.csv(rr$index.treated - 1, paste0(fn, "_rpsm_case_manual.csv"))
write.csv(rr$index.control - 1 - rr$orig.treated.nobs, paste0(fn, "_rpsm_control_manual.csv"))