import torch
import torch.nn as nn
import h5py
import tables
import numpy as np
import pickle
import json
import os
from functools import reduce
from tableone import TableOne
from tqdm import tqdm
import multiprocessing
from functools import partial
import tableone

import matplotlib.pyplot as plt
import seaborn as sns

from data.data_loader import NISDatabase
from data.cohort_builder import CohortBuilder

from utils.experiments import *
from utils.feature_utils import *
from utils.code_mappings import *

from model.autoencoder.autoencoder import AutoEncoder
from model.autoencoder.loss import CustomLoss
from trainer.trainer import Trainer

PSMVARS = ('AGE', 'ELECTIVE', 'FEMALE', 'HCUP_ED', 'HOSP_DIVISION',
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

def main(config_fn, title, sample=None):
    config = json.load(open(config_fn, 'r'))
    config = init(config)

    # Load model
    ae = load_model(config)

    # save for table ones
    tableones = {}

    # Create different cohorts.
    cb, tableones['filtered'], t1records = filter_for_study(config, sample=sample)
    t1records.to_csv(config['global']['PSM_RESULTS'] + 'FilteredDataForPSM.csv')
    # print(tableones['filtered'].tabulate(tablefmt='grid'))

    # Match with DM
    tableones['dm'] = match_with_dm(cb, ae, config)
    # print(tableones['dm'].tabulate(tablefmt='grid'))
    
    try:
        psm_perf_final = np.load(config['global']['PSM_RESULTS'] + 'PSMResults_TTESTS.npy')
        morts = np.load(config['global']['PSM_RESULTS'] + 'PSMResults_MORTS.npy')
        print("Loaded PSM analysis from previous run.")

    except:
        psm_perf_final, morts = analyze_psm_matches(config, t1records)
        morts = np.array(morts).reshape(config['psm']['PSM_END'] - 1, config['psm']['SEEDS'], 2).astype('float')
        np.save(config['global']['PSM_RESULTS'] + 'PSMResults_TTESTS.npy', psm_perf_final)
        np.save(config['global']['PSM_RESULTS'] + 'PSMResults_MORTS.npy', morts)

    manual_name = [config['global']['PSM_RESULTS'] + 'FilteredDataForPSM.csv_rpsm_', '_manual.csv']
    manual_perf, manual_mort = read_psm_file(manual_name, t1records)

    # Mortality
    morts_filt = extract_mortality_from_tab1(tableones['filtered'])
    mort_filt = morts_filt[1] / (morts_filt[0] + 0.01)

    morts_dm = extract_mortality_from_tab1(tableones['dm'])
    mort_dm = morts_dm[1] / (morts_dm[0] + 0.01)
    
    mort_man = manual_mort[1] / (manual_mort[0] + 0.01)

    random_perf, morts_rand = match_with_random_sampling(cb, t1records, config)
    mort_rand = 1 / np.divide(*np.mean(np.array(morts_rand).astype('float'), axis=0))

    ## PLOTTING
    # Mortality
    # plot_mort(morts, config, mort_filt, mort_dm, mort_rand, mort_man, title)
    
    # Heatmap
    psm_perf_heatmap = psm_perf_final[:, :, :psm_perf_final.shape[0] + 1]
    # match_heatmap(psm_perf_heatmap, tableones['dm'], random_perf, manual_perf, config, title)

    # Latent Space
    cb.matchers['deep']['matcher']._embed_in_latent_space()
    cb.matchers['psm'] = {}
    cb.matchers['psm']['matches'] = load_R_psm_results(*manual_name)

    # for matcher, matcher_title in zip():
        # plot_barcodes_in_ls(cb, matcher, matcher_title, config)

    return config, [morts, mort_filt, mort_dm, mort_rand, mort_man], [psm_perf_heatmap, psm_perf_final, tableones['dm'], random_perf, manual_perf], [cb, ['deep', 'psm', 'filtered'], ['DeepMatch', 'Propensity Score Matching', 'All Filtered Patients']]


def filter_for_study(config, sample=None):
    # Create cohort builder
    cb = CohortBuilder(filter_params=config['study']['cohort_build_info'], datasets=config['data'])
    cb.filter_patients(sample=sample)

    # Create table 1 from filtered data.
    t1records, t1t = create_table_one(cb.inds['filtered'], config)
    t1t.to_excel(config['global']['STUDY'] + 'TableOne_Filtered.xlsx')

    t1records['CASECONTROL'] = t1records['CASECONTROL'].replace({'case' : 1, 'control' : 0})
    t1records.to_csv(config['global']['STUDY'] + 'FilteredDataForPSM.csv')

    return cb, t1t, t1records

def match_with_dm(cb, ae, config):
    cb.match_patients(ae, device=config['global']['DEVICE'], save_dir=config['global']['STUDY'])

    cb.set_match_keys('deep')
    t2dat, tab1 = create_table_one(cb.inds['matched'], config)

    tab1.to_excel(config['global']['STUDY'] + 'TableOne_DeepMatch.xlsx')

    return tab1

def match_with_random_sampling(cb_p, t1records_p, config_p, seeds=500):
    
    global cb, t1records, config
    tables.file._open_files.close_all()
    
    cb = cb_p
    t1records = t1records_p
    config = config_p

    with multiprocessing.Pool(config['tableone']['ncpu']) as pool: 
    # with multiprocessing.Pool(20) as pool:
        out = list(tqdm(pool.imap(match_randomly, np.arange(0, 500)), total=seeds, position=0, leave=True))

    # out = []
    # for seed in range(500):
    #     out.append(match_randomly(seed))
    #     print(seed)

    morts = [i[1] for i in out]
    pvals = [i[0] for i in out]

    return pvals, morts

def match_randomly(seed):
    inds = {}
    inds['case'] = cb.inds['filtered']['case']
    inds['control'] = np.random.choice(cb.inds['filtered']['control'], size=cb.inds['filtered']['case'].shape[0], replace=False)
    
    _, tab1 = create_table_one(inds, config)
    
    k = 0
    pvals = []
    mort = extract_mortality_from_tab1(tab1)
    # features = []
    
    for name, pval in tab1.tableone['Grouped by CASECONTROL']['P-Value'].replace({'<0.001' : 0.001}).iteritems():
        if 'CASECONTROL' in name[0]:
            continue

        if 'CM_' in name[0]:
            continue

        if 'Peripheral Vascular Disease' in name[0]:
            continue

        if 'APRDRG' in name[0]:
            break

        if pval != '' and name[0].split(',')[0] in PSMVARS:
            pvals.append(float(pval))
            k += 1

    return [pvals, mort]


def analyze_psm_matches(config, t1records_p):

    global t1records
    t1records = t1records_p

    psm_st = config['psm']['PSM_START']
    psm_end = config['psm']['PSM_END']
    seeds = config['psm']['SEEDS']
    feats = config['psm']['FEATURES']

    prefix = config['global']['PSM_RESULTS'] + 'FilteredDataForPSM.csv' + '_rpsm_'
    # prefix = config['global']['PSM_RESULTS'] + 'NIS_cc_stroke_psminput_20200526.csv' + '_rpsm_'
    names = [[prefix, f'_{pnum}_seed_{seed}.csv'] for pnum in range(psm_st, psm_end+1) for seed in range(1, seeds+1)]
    print(len(names))

    # Clear out TQDM cache
    try:
        list(getattr(tqdm, '_instances'))
        for instance in list(tqdm._instances):
            tqdm._decr_instances(instance)
    except:
        pass

    with multiprocessing.Pool(config['tableone']['ncpu']) as pool:
        psm_performance_flat_all = list(tqdm(pool.imap(_read_psm_file, names), total=len(names), position=0, leave=True))

    morts = [i[1] for i in psm_performance_flat_all]
    psm_performance_flat = [i[0] for i in psm_performance_flat_all]

    psm_perf_final = np.array(psm_performance_flat).reshape(psm_end - psm_st + 1, seeds, len(psm_performance_flat[0]))
    return psm_perf_final, morts
    

def read_psm_file(name, t1records, return_tab=False):
    prefix, suffix = name
    psm_i_res = load_R_psm_results(prefix, suffix)

    psm_i_res['control'] += psm_i_res['case'].shape[0]
    psm_i_rec, tab1 = create_table_one_for_psm(psm_i_res, t1records)

    k = 0
    pvals = []
    mort = extract_mortality_from_tab1(tab1)
    # features = []
    
    for name, pval in tab1.tableone['Grouped by CASECONTROL']['P-Value'].replace({'<0.001' : 0.001}).iteritems():
        if 'CASECONTROL' in name[0]:
            continue

        if 'CM_' in name[0]:
            continue

        if 'Peripheral Vascular Disease' in name[0]:
            continue

        if 'APRDRG' in name[0]:
            break

        if pval != '':
            pvals.append(float(pval))
            # features.append(name[0].split(','))
            k += 1
            
    # return pvals, features
    if return_tab:
        return pvals, mort, tab1
    else:
        return pvals, mort

def extract_mortality_from_tab1(tab1):

    try:
        mort_cont = float(tab1.tableone['Grouped by CASECONTROL']['control']['DIED, n (%)']['1.0'].split('(')[1].split(')')[0])
        mort_case = float(tab1.tableone['Grouped by CASECONTROL']['case']['DIED, n (%)']['1.0'].split('(')[1].split(')')[0])
    except:
        mort_cont = 100 - float(tab1.tableone['Grouped by CASECONTROL']['control']['DIED, n (%)']['0.0'].split('(')[1].split(')')[0])
        mort_case = 100 - float(tab1.tableone['Grouped by CASECONTROL']['case']['DIED, n (%)']['0.0'].split('(')[1].split(')')[0])

    return [mort_cont, mort_case]

def _read_psm_file(name, return_tab=False):
    
    prefix, suffix = name
    psm_i_res = load_R_psm_results(prefix, suffix)

    psm_i_res['control'] += psm_i_res['case'].shape[0]
    psm_i_rec, tab1 = create_table_one_for_psm(psm_i_res, t1records)

    k = 0
    pvals = []

    # mort = [tab1.tableone['Grouped by CASECONTROL']['control']['DIED, n (%)']['1.0'].split('(')[1].split(')')[0], \
            # tab1.tableone['Grouped by CASECONTROL']['case']['DIED, n (%)']['1.0'].split('(')[1].split(')')[0]]
    # features = []

    mort = extract_mortality_from_tab1(tab1)
    
    for name, pval in tab1.tableone['Grouped by CASECONTROL']['P-Value'].replace({'<0.001' : 0.001}).iteritems():
        if 'CASECONTROL' in name[0]:
            continue

        if 'CM_' in name[0]:
            continue

        if 'Peripheral Vascular Disease' in name[0]:
            continue

        if 'APRDRG' in name[0]:
            break

        if pval != '':
            pvals.append(float(pval))
            # features.append(name[0].split(','))
            k += 1
            
    # return pvals, features
    if return_tab:
        return pvals, mort, tab1
    else:
        return pvals, mort


def init(config):

    os.chdir(config['global']['PROJECT_DIR'])
    
    # Add full path to datasets
    for dataset, location in config['data'].items():
        config['data'][dataset] = config['global']['PROJECT_DIR'] + config['global']['DATA_FOLDER'] + location

    return config

def load_model(config):
    """Create things needed for model loading and load the model."""
    device = torch.device(config['global']['DEVICE'])
    db = NISDatabase(config['data']['model'], 'TRAIN')
    
    DEFAULT_BUILD = config['autoencoder']['DEFAULT_BUILD']

    FEATURE_REPRESENTATIONS = {}
    FEATURE_REPRESENTATIONS['embedding'] = config['autoencoder']['EMBEDDING_DICTIONARY']
    FEATURE_REPRESENTATIONS['one_hots'] = config['autoencoder']['ONE_HOTS']
    FEATURE_REPRESENTATIONS['continuous'] = config['autoencoder']['CONTINUOUS']

    DEFAULT_BUILD['features'] = FEATURE_REPRESENTATIONS
 
    find_nlike_features(db.headers, FEATURE_REPRESENTATIONS['embedding'])
    FEATURE_REPRESENTATIONS['one_hots'] = create_onehot_info(db, FEATURE_REPRESENTATIONS['one_hots'], FEATURE_REPRESENTATIONS['embedding'])
    
    DEFAULT_BUILD['features'] = FEATURE_REPRESENTATIONS
    calc_output_dims(DEFAULT_BUILD)

    ae = AutoEncoder(DEFAULT_BUILD).to(device)
    ae.load_state(config['autoencoder']['STATE'], device=device)

    return ae

def create_table_one(inds_dict, config):
    desired_cols = config['tableone']['desired_cols']
    
    dbp = NISDatabase(config['data']['full'], 'case', state_inds=inds_dict, pin_memory=False)
    dbp.set_batch_size(100000)

    records = pd.DataFrame(columns=desired_cols)
    DESIRED_INDS = [find_feature(dbp.headers, feature)[0] for feature in desired_cols]

    outcome_headers = config['tableone']['outcome_cols']

    for cohort in dbp.state_inds.keys():

        dbp.change_state(cohort)
        dbp.set_dataset_key('dataset')

        cohort_data = retrieve_elements_from_db(dbp, DESIRED_INDS, num_cols='multiple')
        cohort_data = pd.DataFrame(cohort_data, columns=desired_cols).astype('int')

        comorbs = get_comorbidities(dbp, config['study']['comorbidities'])
        cohort_data = cohort_data.join(comorbs, how='left')
        
        severities = get_severity(dbp, config['study']['multilevel_comorbidities'])
        cohort_data = cohort_data.join(severities, how='left')

        cohort_data['CHADS_VASC'] = get_chadsvasc(dbp).astype('int')
        cohort_data['CASECONTROL'] = cohort

        dbp.set_dataset_key('outcomes')

        outcomes = retrieve_elements_from_db(dbp, np.arange(0, len(outcome_headers)), num_cols='multiple')
        outcomes = pd.DataFrame(outcomes, columns=outcome_headers).astype('int')
        outcomes.loc[outcomes['DIED'] == -128, 'DIED'] = 0

        record = cohort_data.join(outcomes, how='left')
        records = pd.concat((records, record))
        
        # print(f'{cohort} finished.')
    
    dbp.h5_file.close()

    continuous = ['AGE', 'TOTCHG', 'LOS', 'APRDRG_Risk_Mortality']
    drop = ['APRDRG', 'DRG']

    records.drop(drop, axis=1, inplace=True, errors='ignore')
    categoricals = list(np.setdiff1d(list(records.columns), continuous))

    # print("Finished constructing the table. Now performing stats...")
    table = TableOne(records, columns=list(records.columns), categorical=categoricals, groupby='CASECONTROL', missing=False, pval=True, overall=False)

    for idx in table.tableone.index:
        var_name = idx[0].split(',')[0]
        if var_name in continuous:
            records[var_name] = records[var_name].astype('float')
            var_desc = records.groupby('CASECONTROL')[var_name].describe(include='all')
            table.tableone.loc[f'{var_name}, mean (SD)', ('Grouped by CASECONTROL', 'case')] = '{0:0.1f} ({1:0.1f})'.format(var_desc['mean']['case'], var_desc['std']['case'])
            table.tableone.loc[f'{var_name}, mean (SD)', ('Grouped by CASECONTROL', 'control')] = '{0:0.1f} ({1:0.1f})'.format(var_desc['mean']['control'], var_desc['std']['control'])

    return records, table

def load_R_psm_results(prefix, suffix):
    psm_inds = {'case' : None, 'control' : None}

    for cohort in ['case', 'control']:
        psm_res = pd.read_csv(prefix + f'{cohort}' + suffix)
        psm_res = np.array(psm_res['x'])
        psm_res_unique, counts = np.unique(psm_res, return_counts=True)
        psm_inds[cohort] = psm_res_unique.astype('int')
        
    return psm_inds

def create_table_one_for_psm(psm_res_i, t1records):
    
    records = pd.DataFrame()
    for cohort_type, cohort_inds in psm_res_i.items():
        cohort_data = t1records.iloc[cohort_inds, :].copy()
        cohort_data['CASECONTROL'] = cohort_type
        records = pd.concat((records, cohort_data))
    
    continuous = ['AGE', 'TOTCHG', 'LOS', 'APRDRG_Risk_Mortality']

    categoricals = list(np.setdiff1d(list(records.columns), continuous))
    table = TableOne(records, columns=list(records.columns), categorical=categoricals, groupby='CASECONTROL', missing=False, pval=True, overall=False)

    for idx in table.tableone.index:
        var_name = idx[0].split(',')[0]
        if var_name in continuous:
            records[var_name] = records[var_name].astype('float')
            var_desc = records.groupby('CASECONTROL')[var_name].describe(include='all')
            table.tableone.loc[f'{var_name}, mean (SD)', ('Grouped by CASECONTROL', 'case')] = '{0:0.1f} ({1:0.1f})'.format(var_desc['mean']['case'], var_desc['std']['case'])
            table.tableone.loc[f'{var_name}, mean (SD)', ('Grouped by CASECONTROL', 'control')] = '{0:0.1f} ({1:0.1f})'.format(var_desc['mean']['control'], var_desc['std']['control'])
    
    return records, table

def match_heatmap(psm_perf_final, dm_tab1, rand_pvals, man_pvals, config, title):
    plt.style.use('seaborn-ticks')

    fig, ax = plt.subplots(1, figsize=(20, 14))

    # Fix up DM
    k = 0
    dm_pvals = []
    inds_to_keep = []
    features = []
    
    for name, pval in dm_tab1.tableone['Grouped by CASECONTROL']['P-Value'].replace({'<0.001' : 0.001}).iteritems():
        if 'CASECONTROL' in name[0]:
            continue

        if 'CM_' in name[0]:
            continue

        if 'Peripheral Vascular Disease' in name[0]:
            continue

        if 'APRDRG' in name[0]:
            break

        if pval != '' and name[0].split(',')[0] in PSMVARS:
            dm_pvals.append(float(pval))
            features.append(name[0].split(',')[0])
            k += 1
            

    matches = np.sum(psm_perf_final > 0.05, axis=1) # Sum matches (p > 0.05) across all seeds
    asort = np.argsort(-np.sum(matches, axis=0)) # Sort features by those that were matched most commonly
    matches = matches[:, asort]

    matches = np.vstack(((np.sum(np.array(rand_pvals) > 0.05, axis=0))[asort], matches))
    matches = np.vstack((matches, (np.array(man_pvals)[asort] > 0.05) * 500))
    matches = np.vstack((matches, (np.array(dm_pvals)[asort] > 0.05) * 500))
    feature_names = np.array(features)[asort]

    psm_perf_match = pd.DataFrame(matches.T, index=feature_names, columns=['R', ] + np.arange(2, len(features) + 1).tolist() + ['PSM', 'DM'])
    b1 = sns.heatmap(psm_perf_match, ax=ax, annot=True, cbar=False, fmt='d')

    b1.axes.set_title(f'PSM Heatmap: {title}', fontsize=24, fontweight='bold', pad=12)

    b1.tick_params(labelsize=16)

    fig.tight_layout()

    plt.savefig(config['global']['STUDY'] + 'Heatmap.png')

    return b1

def plot_mort(morts_psm, config, mort_filt, mort_dm, mort_random, mort_manual, title):
    seeds = config['psm']['SEEDS']
    psm_n = config['psm']['PSM_END']
    
    tll = 2
    tul = psm_n + 1
    
    x = np.meshgrid(np.arange(seeds), np.arange(tll, tul))[1].flat[:]
    y = (morts_psm[:, :, 1] / (morts_psm[:, :, 0] + 0.01)).flat[:]
        
    plt.style.use('seaborn-ticks')
                                                           
    fig, ax = plt.subplots(figsize=(14, 6))
    lm = sns.lineplot(x, y, markers=True, ax=ax, ci='sd', err_kws={'interpolate': True})
    ax.plot([tll-2, tll-1, tul, tul + 1], [mort_filt, mort_random, mort_dm, mort_manual], 'bo', markersize=10, markeredgewidth=0, alpha=0.5)
    
    lm.axes.set_title(f'Odds Ratio for Mortality: {title}', fontsize=24, fontweight='bold', pad=12)
    lm.tick_params(labelsize=14)
    
    labels = ['UM', 'R'] + list(np.arange(tll, tul)) + ['DM', 'PSM']
    lm.set_xticks(np.arange(tll - 2, tul + 2))
    lm.set_xticklabels(labels)
    lm.set_xlim(tll - 3, tul + 3)
    
    fig.tight_layout()

    plt.savefig(config['global']['STUDY'] + 'MortalityFigure.png')
    
    return ax

def plot_barcodes_in_ls(cb, matcher_name, title, config):
    fig, axes = plt.subplots(3, 2, figsize=(20, 6))
    
    ls_db = cb.matchers['deep']['matcher'].ls_db
    ls_db.__getitem__(0)
    
    if matcher_name == 'filtered':
        break_ind = ls_db.state_inds['control'].min()
        cases_ls = ls_db.dataset[:break_ind]
        control_ls = ls_db.dataset[break_ind:]
        
    else:
        cases_ls = ls_db.dataset[cb.matchers[matcher_name]['matches']['case'], :]
        control_ls = ls_db.dataset[np.sort(cb.matchers[matcher_name]['matches']['control'] + 
                                              ls_db.state_inds['control'].min()), :]
    
    def barcode(array, size=10):
        return np.multiply(array.reshape(-1, 1), np.ones((1, size))).T

    def barcode_std(arr, size=10):
        array = np.std(arr, axis=0)
        return np.multiply(array.reshape(-1, 1), np.ones((1, size))).T


    
    cases_ls_mean = np.mean(cases_ls, axis=0)
    control_ls_mean = np.mean(control_ls, axis=0)
    
    plt.style.use('seaborn-white')

    axes[0][0].imshow(barcode(cases_ls_mean) > 0, vmin=0, vmax=1)
    axes[0][0].set_title('Mean Cases Barcode, Binary')
    axes[1][0].imshow(barcode(control_ls_mean) > 0, vmin=0, vmax=1)
    axes[1][0].set_title('Mean Controls Barcode, Binary')
    axes[2][0].imshow(((barcode(cases_ls_mean) > 0) * ~(barcode(control_ls_mean) > 0)), vmin=0, vmax=1)
    axes[2][0].set_title('Difference of Means Barcode, Binary')
    
    im = axes[0][1].imshow(barcode(cases_ls_mean), vmin=-2, vmax=2)
    axes[0][1].set_title('Mean Cases Barcode, Continuous')
    axes[1][1].imshow(barcode(control_ls_mean), vmin=-2, vmax=2)
    axes[1][1].set_title('Mean Controls Barcode, Continuous')
    axes[2][1].imshow(np.abs(barcode(cases_ls_mean) - barcode(control_ls_mean)), vmin=-0, vmax=2)
    axes[2][1].set_title('Difference of Means Barcode, Continuous')
    
    for axis in axes.ravel():
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
    
    fig.suptitle('Matching Performance: {0}'.format(title), fontsize=16)

    plt.savefig(config['global']['STUDY'] + 'LatentSpace.png')
    

if __name__ == '__main__':
    # argparse stuff
    
    config = json.load(parser.config)
    main(config)
