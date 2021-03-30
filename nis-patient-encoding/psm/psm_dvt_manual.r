# Load matching library.
# install.packages(c("Matching"))
if(!require(Matching)) install.packages("Matching",repos = "http://cran.us.r-project.org")
if(!require(foreach)) install.packages("foreach",repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel",repos = "http://cran.us.r-project.org")
library(Matching)
library(foreach)
library(doParallel)

# Data loading
fn = "/home/aisinai/work/repos/nis_patient_encoding/study_type/dvt_intervention_10/psm_result/FilteredDataForPSM.csv"
data = read.csv(fn, stringsAsFactors=FALSE)

data$CASECONTROL[data$CASECONTROL == 'case'] <- as.logical(TRUE)
data$CASECONTROL[data$CASECONTROL == 'control'] <- as.logical(FALSE)

data$CASECONTROL <- as.logical(data$CASECONTROL)

# Separate out
treatment <- data$CASECONTROL
# treatment = data$CHRONB01

# Identify all features that we can match on.
# age, sex, race, proximal DVT vs. not, hx of DVT/PE, risk factors for DVT (surgery, cast immobilization, hospitalization, childbirth), use of anticoagulation before DVT, cancer, hypercoagulable states
all_features = c(
        'AGE', 'FEMALE', 'CM_OBESE', 
        'Peripheral Vascular Disease', 
        'Smoking',
        'Chronic Coagulopathy',
        'Cancer',
        'Proximal DVT',
        'History of DVT',
        'History of AC Use')

all_features = make.names(all_features)

features_to_use = all_features
match_features = data[, features_to_use]

print(features_to_use)

match_features[] <- lapply(match_features, as.numeric)

rr = Match(Tr=treatment, X=match_features, M=1, ties=FALSE, replace=FALSE)

write.csv(rr$index.treated - 1, paste0(fn, "_rpsm_case_manual.csv"))
write.csv(rr$index.control - 1 - rr$orig.treated.nobs, paste0(fn, "_rpsm_control_manual.csv"))