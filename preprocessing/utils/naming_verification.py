import re

def tight_verify_name(name_to_verify, name_range):
    for possible_sequence_name in name_range:
        # allow for variations in sequence names with sequence Id at the end
        spc_ct_name_regex = possible_sequence_name + '(| ([0-9]|[1-9][0-9]|[1-9][0-9][0-9]))$'
        if re.match(spc_ct_name_regex, name_to_verify):
            return True

def loose_verify_name(name_to_verify, name_range):
    for possible_sequence_name in name_range:
        if possible_sequence_name in name_to_verify:
            return True
