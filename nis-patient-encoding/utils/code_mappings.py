import numpy as np
import pandas as pd

# def convert_icd9_to_hcup(code):
#     code = str(code)
#     code_split = code.split('.')
#     code_pre = reduce(sum, ['0' for i in range(0, 2 - len(code_split[0]))]) + code_pre
#     code_post = 

def map_missing_codes(series):
    """ Replaces missing values with -127 in the series.
    
    Arguments:
        series: 
    
    """
    return series.apply(lambda cell: MISS_VAL_FILL if (pd.isna(cell) or (cell in MISSING_VALS)) else cell)


def map_icd9_codes(series):
    """ Maps alphanumeric ICD-9 codes to purely numeric, notably of type int32.
    
    Wrapper that takes a pandas Series containing ICD-9 codes and maps it based on the following logic:
    000-999 => 0 - 99,999 (should be this way already)
    V00-V99 => 100,000 - 109,999
    E000-E999 => 200,000 - 299,999 (likely 299,99).
    
    Arguments:
        series: Pandas data series containing ICD-9 codes. May be of type object.
        
    Returns:
        series: Pandas data series containing aforementioned mapping, of type int32.
        
    """

    return series.apply(map_icd9_to_numeric)

def map_icd9_to_numeric(code):
    """ 
    """

    code_str = str(code)

    if code_str[0] == 'V':
        len_code = len(code_str[1:])
        scale = 10**(4 - len_code)
        base = 100000
        icd9_nums = int(float(code_str[1:]))

    elif code_str[0] == 'E':
        len_code = len(code_str[1:])
        scale = 10**(5 - len_code)
        base = 200000
        icd9_nums = int(float(code_str[1:]))

    else:
        len_code = len(code_str[0:])
        scale = 10**(5 - len_code)
        base = 0
        icd9_nums = int(float(code_str[0:]))

    code_mapped = base + scale * icd9_nums

    return code_mapped


def map_mccs_codes(series):
    """
    ab.cd.ef.gh => a*10^7 + b*10^6 + c*10^5 + d*10^4 + e*10^3 + f*10^2 + g*10^1 + h*10^0
    """
    
    return series.apply(map_mccs_to_numeric)

def map_mccs_to_numeric(code):
    if is_missing(code):
        return MISS_VAL_FILL
    
    code_str = str(code)
    groups = code_str.split('.')
    code_flat = ''
    for group in groups: 
        code_flat += group

    code_map = 0
    for i, val in enumerate(code_flat):
        scale = 10**(7 - i)
        code_map += int(float(val)) * scale

    return code_map

def map_ecode_to_numeric(code):
    if is_missing(code):
        return MISS_VAL_FILL
    else:
        return int(float(code[1:]))

def map_ecode_codes(series):
    """
    """
    
    return series.apply(map_ecode_to_numeric)