from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, InListValidation, DateFormatValidation

# FINAL PARAMETERS
# parameters used as input for the statistical model
# and their validation schema
# all parameters have to be numerical

# the ORDER of these parameters is primordial as it defines the order of input into xgboost

parameters = [
    'age',
    'sex',
    'height',
    'weight',
    ## timing parameters
    # dummies for onset_known
    # (onset_known_no is dropped as it's encoding can be deduced from onset_known_yes)
    'onset_known_yes',
    'onset_known_wake_up',
    # onset to ct in minutes
    'onset_to_ct',
    # clinical parameters
    'NIH admission',
    'bp_syst',
    'bp_diast',
    'glucose', # not essential
    'créatinine', # not essential
    # cardiovascular risk factors
    'hypertension',
    'diabetes',
    'hyperlipidemia',
    'smoking',
    'atrialfib',
    # ATCDs
    'stroke_pre',
    'tia_pre',
    'ich_pre',
    # general Treatment
    'treat_antipatelet',
    'treat_anticoagulant',
    # acute treatment
    'treat_ivt_before_ct', # ie ivt was started before imaging
]

validation_schema = Schema([
    Column('age', [InRangeValidation(1, 120)]),
    Column('sex', [InListValidation([0, 1])]),
    Column('height', [InRangeValidation(50, 300)]),
    Column('weight', [InRangeValidation(10, 400)]),

    Column('onset_known_yes', [InListValidation([0, 1])]),
    Column('onset_known_wake_up', [InListValidation([0, 1])]),

    Column('onset_to_ct', [InRangeValidation(0, 43800)]), # onset to CT time max is 1 month

    Column('NIH admission', [InRangeValidation(0, 43)]),
    Column('bp_syst', [InRangeValidation(0, 300)]),
    Column('bp_diast', [InRangeValidation(0, 300)]),
    Column('glucose', [InRangeValidation(0.1, 30)]),
    Column('créatinine', [InRangeValidation(0.1, 1000)]),

    Column('hypertension', [InListValidation([0, 1])]),
    Column('diabetes', [InListValidation([0, 1])]),
    Column('hyperlipidemia', [InListValidation([0, 1])]),
    Column('smoking', [InListValidation([0, 1])]),
    Column('atrialfib', [InListValidation([0, 1])]),

    Column('stroke_pre', [InListValidation([0, 1])]),
    Column('tia_pre', [InListValidation([0, 1])]),
    Column('ich_pre', [InListValidation([0, 1])]),

    Column('treat_antipatelet', [InListValidation([0, 1])]),
    Column('treat_anticoagulant', [InListValidation([0, 1])]),

    Column('treat_ivt_before_ct', [InListValidation([0, 1])]),
])
