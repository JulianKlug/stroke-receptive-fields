from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, InListValidation, DateFormatValidation

# INPUT PARAMETERS
# parameters used to import from DB for calculation of final parameters
# and the validation schema of the inputs

parameters = [
    'age',
    'sex',
    'height',
    'weight',
    # timing parameters
    'onset_known',
    # parameters needed to calculate onset to ct
    'onset_time',
    'Firstimage_date',
    # clinical parameters
    'NIH admission',
    'bp_syst',
    'bp_diast',
    'glucose', # to discuss
    'créatinine', # to discuss
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
    # Treatment
    'treat_antipatelet',
    'treat_anticoagulant',
    'ivt_start', # needed to check if started before imaging
    'treat_ivt',
    'iat_start', # needed to check if started before imaging
]

validation_schema = Schema([
    Column('age', [InRangeValidation(1, 120)]),
    Column('sex', [InListValidation(['m', 'f'])]),
    Column('height', [InRangeValidation(50, 300)]),
    Column('weight', [InRangeValidation(10, 400)]),

    Column('onset_known', [InListValidation(['yes', 'no', 'wake_up'])]),
    Column('Firstimage_date', [DateFormatValidation('%Y-%m-%d %H:%M:%S')]),
    Column('onset_time', [DateFormatValidation('%Y-%m-%d %H:%M:%S')]),

    Column('NIH admission', [InRangeValidation(0, 43)]),
    Column('bp_syst', [InRangeValidation(0, 300)]),
    Column('bp_diast', [InRangeValidation(0, 300)]),
    Column('glucose', [InRangeValidation(0.1, 30)]),
    Column('créatinine', [InRangeValidation(0.1, 1000)]),

    Column('hypertension', [InListValidation(['yes', 'no'])]),
    Column('diabetes', [InListValidation(['yes', 'no'])]),
    Column('hyperlipidemia', [InListValidation(['yes', 'no'])]),
    Column('smoking', [InListValidation(['yes', 'no'])]),
    Column('atrialfib', [InListValidation(['yes', 'no'])]),

    Column('stroke_pre', [InListValidation(['yes', 'no'])]),
    Column('tia_pre', [InListValidation(['yes', 'no'])]),
    Column('ich_pre', [InListValidation(['yes', 'no'])]),

    Column('treat_antipatelet', [InListValidation(['yes', 'no'])]),
    Column('treat_anticoagulant', [InListValidation(['yes', 'no'])]),
    Column('ivt_start', [DateFormatValidation('%Y-%m-%d %H:%M:%S')]),
    Column('treat_ivt', [InListValidation(['yes', 'no', 'started_before_admission'])]),
    Column('iat_start', [DateFormatValidation('%Y-%m-%d %H:%M:%S')]),
])
