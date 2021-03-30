import numpy as np

def find_nlike_features(headers, emb_dict):
    """
    """

    headers = [header.decode('utf-8') for header in headers if isinstance(header, bytes)]

    for emb_name, emb in emb_dict.items():
        # idenitfy feature vector
        emb_dict[emb_name]['feature_idx'] = []
        len_name = len(emb['header_prefix'])

        if isinstance(emb['header_prefix'], bytes):
            emb['header_prefix'] = emb['header_prefix'].decode('utf-8')

        for idx, header in enumerate(headers):
            if header[:len_name] == emb['header_prefix'] and header[len_name] in [str(i) for i in range(0, 10)]:
                emb_dict[emb_name]['feature_idx'].append(idx)


def create_onehot_info(db, one_hots, emb_dict):
    for idx, header in enumerate(db.headers):
        still_good = True
        # if idx == 0:
        #     still_good = False
            
        if header[:5] == b'CHRON':
            still_good = False

        for embed_vec in emb_dict.values():
            if idx in embed_vec['feature_idx']:
                still_good = False
                break

        if header[:2] == b'DX':
            still_good = False

        if still_good:
            one_hots[header.decode('utf-8')]['feature_idx'] = idx

    return one_hots


def feature_types_for_loss(build):
    loss_params = { 'continuous' : [],
                    'categorical' : [],
                    }
    for feature_type, ft_info in build.items():
        if feature_type == 'continuous':
            loss_params['continuous'].extend([idx for idx in ft_info.values()]) # idx
        elif feature_type == 'one_hots':
            cat_idx = [cat_info['idx'] for cat_info in ft_info.values()]
            loss_params['categorical'].extend(cat_idx)
        elif feature_type == 'embedding':
            emb_idx = []
            for emb_ft_info in ft_info.values():
                emb_idx.extend(emb_ft_info['features'])
            loss_params['categorical'].extend(emb_idx)

    return loss_params
