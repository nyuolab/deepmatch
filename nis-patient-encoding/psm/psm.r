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
all_features = c('AGE', 'ELECTIVE', 'FEMALE', 'HCUP_ED', 'HOSP_DIVISION',
       'PAY01', 'PL_NCHS', 'TRAN_IN', 'TRAN_OUT', 'ZIPINC_QRTL',
       'Essential Hypertension', 'Hypertension - Complicated', 'Dyslipidemia',
       'Atrial Fibrillation', 'Coronary Artery Disease', 'Diabetes', 'Smoking',
       'Prior MI', 'Prior PCI', 'Prior CABG', 'Family History of CAD',
       'Chronic Cardiac Disease', 'Chronic CHF', 'Chronic Stroke',
       'Cardiac Conduction Disorders', 'Chronic Respiratory Disease', 'Cancer',
       'Chronic Fluid + Electrolyte Disorders', 'Chronic Anemia',
       'Chronic Coagulopathy', 'Chronic Neurological Conditions',
       'Chronic GI Illnesses', 'Chronic Hepatobiliary Disease',
       'Chronic Kidney Disease', 'Immune-mediated Rheumatopathies',
       'Osteoporosis', 'Osteoarthritis', 'Skin Disorders',
       'Genitourinary Disorders', 'Multilevel Liver Disease',
       'Multilevel Diverticulitis', 'Multilevel Chronic Kidney Disease',
       'Multilevel Diabetes', 'CHADS_VASC')

all_features = make.names(all_features)

registerDoParallel(60)

for (i in 2:length(all_features)) {
       foreach (seed=1:500) %dopar% { 
              # feature_index = sample(1:length(all_features), i, replace=FALSE)
              # features_to_use = all_features[1:i]
              
              features_to_use = sample(all_features, i, replace=FALSE)
              match_features = data[, features_to_use]

              print(paste('Now operating on ', i, ' features.'))

              match_features[] <- lapply(match_features, as.numeric)

              rr = Match(Tr=treatment, X=match_features, M=1, ties=FALSE, replace=FALSE)

              write.csv(rr$index.treated - 1, paste0(fn, "_rpsm_case_", i, "_seed_", seed, ".csv"))
              write.csv(rr$index.control - 1 - rr$orig.treated.nobs, paste0(fn, "_rpsm_control_", i, "_seed_", seed, ".csv"))
              print(paste0(fn, "_rpsm_control_", i, "_seed_", seed, ".csv"))
       }
	
}
