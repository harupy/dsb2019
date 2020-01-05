from zlib import crc32


def hash_mod(x, n_splits):
    return (crc32(x.encode('utf-8')) & 0xffffffff) % n_splits


def hash_mod_sample(df, col, n_splits, keep=[0]):
    hm = df[col].map(lambda x: hash_mod(x, n_splits))
    return df[hm.isin(keep)]
