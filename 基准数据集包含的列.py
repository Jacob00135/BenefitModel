feature_columns = [
    'RID', 'VISCODE', 'filename', 'ADAS13', 'COG',
    'age', 'gender', 'education', 'trailA',
    'trailB', 'boston', 'digitB', 'digitBL', 'digitF', 'digitFL', 'animal',
    'gds', 'lm_imm', 'lm_del', 'mmse', 'npiq_DEL', 'npiq_HALL', 'npiq_AGIT',
    'npiq_DEPD', 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR',
    'npiq_MOT', 'npiq_NITE', 'npiq_APP', 'faq_BILLS', 'faq_TAXES',
    'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE', 'faq_MEALPREP', 'faq_EVENTS',
    'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL', 'his_NACCFAM', 'his_CVHATT',
    'his_CVAFIB', 'his_CVANGIO', 'his_CVBYPASS', 'his_CVPACE', 'his_CVCHF',
    'his_CVOTHR', 'his_CBSTROKE', 'his_CBTIA', 'his_SEIZURES', 'his_TBI',
    'his_HYPERTEN', 'his_HYPERCHO', 'his_DIABETES', 'his_B12DEF', 'his_THYROID',
    'his_INCONTU', 'his_INCONTF', 'his_DEP2YRS', 'his_DEPOTHR', 'his_PSYCDIS',
    'his_ALCOHOL', 'his_TOBAC100', 'his_SMOKYRS', 'his_PACKSPER',
    'his_ABUSOTHR'
]

stat_columns = [
    'COG', 'education', 'age', 'gender'
]


if __name__ == '__main__':
    print(feature_columns)
