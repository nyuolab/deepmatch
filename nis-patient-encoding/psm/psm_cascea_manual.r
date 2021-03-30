# Load matching library.
# install.packages(c("Matching"))
if(!require(Matching)) install.packages("Matching",repos = "http://cran.us.r-project.org")
if(!require(foreach)) install.packages("foreach",repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel",repos = "http://cran.us.r-project.org")
library(Matching)
library(foreach)
library(doParallel)

# Data loading
fn = "/home/aisinai/work/repos/nis_patient_encoding/study_type/cas_cea_10/psm_result/FilteredDataForPSM.csv"
data = read.csv(fn, stringsAsFactors=FALSE)

data$CASECONTROL[data$CASECONTROL == 'case'] <- as.logical(TRUE)
data$CASECONTROL[data$CASECONTROL == 'control'] <- as.logical(FALSE)

data$CASECONTROL <- as.logical(data$CASECONTROL)

# Separate out
treatment <- data$CASECONTROL
# treatment = data$CHRONB01

# Identify all features that we can match on.
# age, sex, BMI (obesity), cardiac comorbidities (arrhythmias, previous MI/CAD, carotid disease, cerebrovascular disease, COPD, peripheral vascular disease, a-fib), diabetes
all_features = c(
        'AGE', 'FEMALE', 'CM_OBESE', 
        'Essential Hypertension', 'Hypertension - Complicated', 
        'Dyslipidemia',
        'Coronary Artery Disease',
        'Prior MI', 
        'Prior CABG',
        'Chronic CHF', 
        'Atrial Fibrillation', 
        'Cardiac Conduction Disorders',
        'Chronic Stroke', 
        'Peripheral Vascular Disease',
        'Diabetes', 
        'Smoking')

all_features = make.names(all_features)

features_to_use = all_features
match_features = data[, features_to_use]

print(features_to_use)

match_features[] <- lapply(match_features, as.numeric)

rr = Match(Tr=treatment, X=match_features, M=1, ties=FALSE, replace=FALSE)

write.csv(rr$index.treated - 1, paste0(fn, "_rpsm_case_manual.csv"))
write.csv(rr$index.control - 1 - rr$orig.treated.nobs, paste0(fn, "_rpsm_control_manual.csv"))