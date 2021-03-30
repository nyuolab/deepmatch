import numpy as np
import pandas as pd
import tables

import os
import sys
import re


class NISPreprocessor(object):
    """Preprocess NIS files."""

    def __init__(self, pattern, directory, **kwargs):
        self.pattern = pattern # split on 'Y' for year.
        self.directory = directory

        self.chunk_size = kwargs.get('chunk_size', 100000)

        self.find_all_years()


    def find_all_years(self):
        self.years = {}
        prefix, suffix = self.pattern.split('Y')

        for file in os.listdir(self.directory):
            if prefix in file and suffix in file:
                year = file.split('_')[1]
                self.years[year] = {'file': file}


    def get_column_coverage(self):
        self.column_coverage = pd.DataFrame(dtype='uint8')

        for year, year_info in self.years.items():
            fn = self.directory + year_info['file']
            self.years[year]['columns'] = list(next(pd.read_sas(fn, chunksize=1)).columns)

            col_sn = [col.upper() for col in self.years[year]['columns']]
            cols_y_s = pd.Series(np.ones(len(col_sn), dtype='uint8'), index=col_sn, name=year)

            # Merge columns
            self.column_coverage = self.column_coverage.join(cols_y_s, how='outer')

        self.column_coverage = self.column_coverage[np.sort(self.column_coverage.columns)]

        # Remove columns that are repeating numbers (DX1, DXCCS3, etc.)
        for column in self.column_coverage.index:
            col_split = re.split(r'(-?\d*\.?\d+)', column)
            if col_split[0] == 'I':
                # I, 10, _X, num, blank
                col_split = [''.join(col_split[:1]) + col_split[2][1:], ]
            

    

    @staticmethod
    def convert_sas_to_hdf5(filename):
        nis_i = pd.read_sas(DATA_FOLDER + filename, chunksize=self.chunk_size)
        nis_o = tables.open_file(DATA_FOLDER + filename.split('csv')[0] + '.h5', 'w')

        n_rows = nis_i.row_count
        n_cols = nis_i.row_length

        # Store headers.


        # Create storage array for all data elements.
        nis_o.create_array('/', 'dataset', shape=(n_rows, n_cols), atom=tables.Float64Atom())

        # Concatenate all chunks
        for chunk in f:
            pass


# ########################################################################################################################
# #################################################### HELPER FUNCTIONS ##################################################
# ########################################################################################################################

def is_missing(code):
    return (pd.isna(code) or (code in MISSING_VALS))
    
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
    if is_missing(code):
        return MISS_VAL_FILL

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

def find_schema(series_name):
    """
    """
    if series_name in SCHEMA_DICT.keys():
        schema_key = series_name
        
    elif series_name[:3] == 'CM_':
        schema_key = 'CM_'
        
    else:
        prefix = re.split(r'\d', series_name)[0] # Prefix before #
        schema_key = prefix + 'n'
        
    return schema_key, SCHEMA_DICT[schema_key]