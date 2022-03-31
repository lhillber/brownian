#! user/bin/python3
from copy import deepcopy
from numpy import ndarray, int64
from hashlib import sha1
from json import dumps

def dict_uid(d, uid_keys):
    """
    Create a unique ID for a dict based on the values
    associated with uid_keys.
    """
    name_dict = {}
    dc = deepcopy(d)
    for k, v in dc.items():
        if k in uid_keys:
            name_dict[k] = v
    dict_el_array2list(name_dict)
    dict_el_int2float(name_dict)
    dict_key_to_string(name_dict)
    uid = sha1(dumps(name_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return uid


def dict_el_array2list(d):
    """
    Convert dict values to lists if they are arrays.
    """
    for k, v in d.items():
        if type(v) == ndarray:
            d[k] = list(v)
        if type(v) == dict:
            dict_el_array2list(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_array2list(vel)
                if type(vel) == ndarray:
                    v[i] = list(vel)


def dict_el_int2float(d):
    """
    Convert dict values to floats if they are ints.
    """
    for k, v in d.items():
        if type(v) in (int, int64) :
            d[k] = float(v)
        if type(v) == dict:
            dict_el_int2float(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_int2float(vel)
                if type(vel) == int:
                    v[i] = float(vel)

def dict_key_to_string(d):
    """
    Convert dict keys to strings.
    """
    for k, v in d.items():
        d[str(k)] = v
        if type(k) != str:
            del d[k]
        if type(v) == dict:
            dict_key_to_string(v)
        if type(v) == list:
            for vel in v:
                if type(vel) == dict:
                    dict_key_to_string(vel)
