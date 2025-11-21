import pandas as pd
import numpy as np
import warnings
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.diagnostic import linear_harvey_collier
from statsmodels.stats.contrast import ContrastResults
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1
from scipy.stats import t as student_t
from did_multiplegt_dyn_core import did_multiplegt_dyn_core
from _utils import *

def did_multiplegt_main(
        df,
        outcome,
        group,
        time,
        treatment,
        cluster = None,
        effects = 1,
        placebo = 1,
        normalized = False,
        effects_equal = False,
        controls = None,
        trends_nonparam = None,
        trends_lin = False,
        continuous = 0,
        weight = None,
        predict_het = None,
        same_switchers = False,
        same_switchers_pl = False,
        switchers = "",
        only_never_switchers = False,
        ci_level = 95,
        save_results = None,
        less_conservative_se = False,
        dont_drop_larger_lower = False,
        drop_if_d_miss_before_first_switch = False,
):


    dict_glob = {}
    gr_id = None
    weight_XX = None
    F_g_XX = None
    F_g_trunc_XX = None
    N_gt_XX = None
    T_g_XX = None
    U_Gg_var_global_XX = None
    Yg_Fg_min1_XX = None
    Yg_Fg_min2_XX = None
    avg_diff_temp_XX = None
    avg_post_switch_treat_XX = None
    avg_post_switch_treat_XX_temp = None
    clust_U_Gg_var_global_XX = None
    cluster_XX = None
    cluster_var_g_XX = None
    controls_time_XX = None
    count_time_post_switch_XX = None
    count_time_post_switch_XX_temp = None
    counter = None
    counter_temp = None
    d_F_g_XX = None
    d_F_g_temp_XX = None
    d_fg_XX = None
    d_sq_XX = None
    d_sq_int_XX = None
    d_sq_temp_XX = None
    diff_y_XX = None
    ever_change_d_XX = None
    fd_X_all_non_missing_XX = None
    first_obs_by_clust_XX = None
    first_obs_by_gp_XX = None
    group_XX = None
    last_obs_D_bef_switch_XX = None
    last_obs_D_bef_switch_t_XX = None
    max_time_d_nonmiss_XX = None
    mean_D = None
    mean_Y = None
    min_time_d_miss_aft_ynm_XX = None
    min_time_d_nonmiss_XX = None
    min_time_y_nonmiss_XX = None
    never_change_d_XX = None
    sd_het = None
    sum_weights_control_XX = None
    temp_F_g_XX = None
    time_XX = None
    time_d_miss_XX = None
    time_d_nonmiss_XX = None
    time_y_nonmiss_XX = None
    treatment_XX_v1 = None
    var_F_g_XX = None



        
    # 1. Checking that syntax correctly specified
    if not same_switchers and same_switchers_pl:
        raise ValueError("The same_switchers_pl option only works if same_switchers is specified as well!")

    # Continuous option: checking polynomial order
    if continuous == 0:
        degree_pol = continuous
    else:
        degree_pol = continuous

    
    # Subset columns but do NOT rename originals
    original_names = [outcome, group, time, treatment]
    if trends_nonparam:
        original_names += trends_nonparam
    if weight:
        original_names.append(weight)
    if controls:
        original_names += controls
    if (cluster) and (cluster != group):
        original_names.append(cluster)
    if predict_het:
        original_names += list(predict_het[0])
    print(original_names)
    df = df[original_names].copy()
    # Standardize names
    df = df.rename(columns={
        outcome: 'outcome',
        group:   'group',
        time:    'time',
        treatment: 'treatment'
    })

    # 2. Data preparation steps
    # Patch the cluster variable
    if cluster is not None:
        if cluster == group:
            cluster = None
        else:
            df['cluster_XX'] = df[cluster].copy()
            cluster = 'cluster_XX' 
            
    
    group = 'group'
    outcome = 'outcome'
    time = 'time'
    treatment = 'treatment'

    # Standardize names
    df = df.rename(columns={
        outcome: 'outcome',
        group:   'group',
        time:    'time',
        treatment: 'treatment',
        cluster : 'cluster_XX'
    })
       
    # Drop rows with missing group or time
    df = df.dropna(subset=[group, time]).copy()
    # Drop rows with missing controls
    if controls:
        for var in controls:
            df = df.dropna(subset=[var]).copy()

    # Drop rows with missing controls
    # Drop rows with missing cluster
    if cluster is not None:
        df = df.dropna(subset=['cluster_XX']).copy()

    # # Drop groups with always-missing treatment or outcome
    df['mean_D'] = df.groupby(group)[treatment].transform(lambda x: x.mean(skipna=True))
    df['mean_Y'] = df.groupby(group)[outcome].transform(lambda x: x.mean(skipna=True))
    df = df[df['mean_D'].notna() & df['mean_Y'].notna()]
    df = df.drop(columns=['mean_D', 'mean_Y'])

    # Predict_het option for heterogeneous treatment effects
    # Predict_het option for heterogeneous treatment effects
    predict_het_good = []
    if predict_het is not None:
        if not (isinstance(predict_het, list) and len(predict_het) == 2):
            raise ValueError(
                "Syntax error in predict_het option: list with 2 elements required. "
                "Set the second element to -1 to include all the effects."
            )
        if normalized:
            warnings.warn(
                "The options normalized and predict_het cannot be specified together; "
                "predict_het will be ignored."
            )
        else:
            pred_het, het_effects = predict_het
            for v in pred_het:
                df['sd_het'] = df.groupby(group)[v].transform(lambda x: x.std(skipna=True))
                if float(df['sd_het'].mean()) == 0:
                    predict_het_good.append(v)
                else:
                    warnings.warn(
                        f"The variable {v} specified in predict_het is time-varying; it will be ignored."
                    )
                df = df.drop(columns=['sd_het'])

    # — Collapse and weight —

    # 1. Create the weight variable
    if weight is None:
        df['weight_XX'] = 1
    else:
        df['weight_XX'] = df[weight]
    df['weight_XX'] = df['weight_XX'].fillna(0)

    # 2. Check whether data are already at the (group, time) level
    df['counter_temp'] = 1
    df['counter'] = df.groupby([group, time])['counter_temp'].transform('sum')
    aggregated_data = df['counter'].max() == 1
    df = df.drop(columns=['counter', 'counter_temp'])

    # assume df is your DataFrame, aggregated_data is a bool, cluster is either a column name or None,
    # and trends_nonparam, controls, predict_het_good are lists of column-names (or empty lists)

    if not aggregated_data:
        # zero out weight where treatment is missing
        df.loc[df['treatment'].isna(), 'weight_XX'] = 0

        # if no clustering variable specified, create a dummy
        if cluster is None:
            df['cluster_XX'] = 1

        # build list of columns to re‐aggregate with weighted means
        to_wmean = ['treatment', 'outcome']
        to_wmean += trends_nonparam or []
        if weight is not None:
            to_wmean.append(weight)
        to_wmean += controls or []
        to_wmean += predict_het_good or []
        to_wmean.append('cluster_XX')

        # groupby
        grp = df.groupby(['group', 'time'])

        # df1: weighted means
        df1 = (
            grp
            .apply(lambda g: pd.Series({
                col: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), 'weight_XX'])
                    if g['weight_XX'].sum() > 0 else np.nan
                for col in to_wmean
            }))
            .reset_index()
        )

        # df2: sum of weights
        df2 = grp['weight_XX'].sum().reset_index(name='weight_XX')

        # merge back
        df = pd.merge(df1, df2, on=['group', 'time'] )

        # drop dummy cluster if we added it
        if cluster is None:
            df = df.drop(columns='cluster_XX')



    # — Generate factorized versions of Y, G, T and D —
    # — Generate factorized versions of Y, G, T and D —
    df['outcome_XX'] = df[outcome].copy()
    df = df.sort_values(time)
    df['group_XX']     = pd.factorize(df[group])[0] + 1
    df['time_XX']      = pd.factorize(df[time])[0] + 1
    df['treatment_XX'] = df[treatment].copy()

    # ensure numeric
    df['group_XX'] = df['group_XX'].astype(float)
    df['time_XX']  = df['time_XX'].astype(float)

    # — Variables for imbalanced panels —
    # first/last date where D not missing
    df['time_d_nonmiss_XX'] = np.where(df['treatment_XX'].notna(), df['time_XX'], np.nan)
    # first date where Y not missing
    df['time_y_nonmiss_XX'] = np.where(df['outcome_XX'].notna(), df['time_XX'], np.nan)

    grp = df.groupby('group_XX')
    df['min_time_d_nonmiss_XX'] = grp['time_d_nonmiss_XX'].transform( lambda x: x.min(skipna=True) )
    df['max_time_d_nonmiss_XX'] = grp['time_d_nonmiss_XX'].transform( lambda x: x.max(skipna=True) )
    df['min_time_y_nonmiss_XX'] = grp['time_y_nonmiss_XX'].transform( lambda x: x.min(skipna=True) )

    # first date D missing after Y seen
    df['time_d_miss_XX'] = np.where(
        df['treatment_XX'].isna() & (df['time_XX'] >= df['min_time_y_nonmiss_XX']),
        df['time_XX'],
        np.nan
    )
    grp = df.groupby('group_XX')
    df['min_time_d_miss_aft_ynm_XX'] = grp['time_d_miss_XX'].transform('min')

    # drop intermediate cols
    df = df.drop(columns=['time_d_nonmiss_XX','time_y_nonmiss_XX','time_d_miss_XX'])

    



    # — Baseline treatment D_{g,1} —
    df['d_sq_temp_XX'] = np.where(
        df['time_XX'] == df['min_time_d_nonmiss_XX'],
        df['treatment_XX'],
        np.nan
    )
    grp = df.groupby('group_XX')
    df['d_sq_XX'] = grp['d_sq_temp_XX'].transform('mean')
    df = df.drop(columns=['d_sq_temp_XX'])

    # — Enforce “Design Restriction 2” —

    df['diff_from_sq_XX'] = df['treatment_XX'] - df['d_sq_XX']
    df = df.sort_values(['group_XX','time_XX'])
    T_XX = int(df['time_XX'].max())

    if not dont_drop_larger_lower:  # equivalent to if "`dont_drop_larger_lower'"==""
        # 1. sort like Stata
        df = df.sort_values(['group_XX', 'time_XX'])

        # 2. strict increase
        df['ever_strict_increase_XX'] = ((df['diff_from_sq_XX'] > 0) & df['treatment_XX'].notna()).astype(int)
        df['ever_strict_increase_XX'] = (
            df.groupby('group_XX')['ever_strict_increase_XX']
                .transform(lambda s: s.cumsum().clip(upper=1))
        )

        # 3. strict decrease
        df['ever_strict_decrease_XX'] = ((df['diff_from_sq_XX'] < 0) & df['treatment_XX'].notna()).astype(int)
        df['ever_strict_decrease_XX'] = (
            df.groupby('group_XX')['ever_strict_decrease_XX']
                .transform(lambda s: s.cumsum().clip(upper=1))
        )

        # 4. drop rows where both == 1
        df = df[~((df['ever_strict_increase_XX'] == 1) & (df['ever_strict_decrease_XX'] == 1))]
        print(df.ever_strict_increase_XX.value_counts().reset_index().sort_values('ever_strict_increase_XX'))
        print(df.ever_strict_decrease_XX.value_counts().reset_index().sort_values('ever_strict_decrease_XX'))
        # 5. drop helper columns
        df = df.drop(columns=['ever_strict_increase_XX', 'ever_strict_decrease_XX'])






    # count groups
    # 1. Ever changed treatment
    df['ever_change_d_XX'] = (df['diff_from_sq_XX'].abs() > 0) & df['treatment_XX'].notna()

    # carry forward “ever change” within each group
    df = df.sort_values(['group_XX','time_XX'])
    df['ever_change_d_XX'] = df.groupby('group_XX')['ever_change_d_XX'].cummax()

    
    # 2. First treatment-change date
    df['temp_F_g_XX'] = np.where(
        df['ever_change_d_XX'] & ~df.groupby('group_XX')['ever_change_d_XX'].shift(1).fillna(False),
        df['time_XX'],
        0
    )
    df['F_g_XX'] = df.groupby('group_XX')['temp_F_g_XX'].transform('max')


    
    # 3. Continuous option: polynomials of D_{g,1}
    if continuous > 0:
        for p in range(1, degree_pol + 1):
            p = int(p)
            df[f'd_sq_{p}_XX'] = df['d_sq_XX'] ** p
        df['d_sq_XX_orig'] = df['d_sq_XX']
        df['d_sq_XX'] = 0

    # 4. Integer levels of d_sq_XX
    # df['d_sq_int_XX'] = pd.factorize(df['d_sq_XX'])[0] + 1
    codes = df.groupby('d_sq_XX', sort=True).ngroup()
    df['d_sq_int_XX'] = (codes + 1).where(df['d_sq_XX'].notna(), np.nan)

    # 5. Drop baseline treatments with no variation in F_g
    if trends_nonparam:
        group_cols = ['d_sq_XX'] + trends_nonparam
    else:
        group_cols = ['d_sq_XX']
    df['var_F_g_XX'] = df.groupby(group_cols)['F_g_XX'].transform('std').round(3)

    
    df = df[df['var_F_g_XX'] > 0].drop(columns=['var_F_g_XX'])
    if df.empty:
        raise ValueError(
            "No treatment effect can be estimated. Design Restriction 1 is not satisfied."
        )
    
    # count groups
    G_XX = df['group_XX'].unique().size


    # 6. Restrict to cells with at least one “never-changer”
    df['never_change_d_XX'] = 1 - df['ever_change_d_XX'].astype(int)
    if trends_nonparam:
        ctrl_group = ['time_XX','d_sq_XX'] + trends_nonparam
    else:
        ctrl_group = ['time_XX','d_sq_XX']

    df['controls_time_XX'] = df.groupby(ctrl_group)['never_change_d_XX'].transform('max')
    df = df[df['controls_time_XX'] > 0]


    # 7. Adjust F_g for never-changers
    t_min_XX = df['time_XX'].min()
    T_max_XX = df['time_XX'].max()
    df.loc[df['F_g_XX'] == 0, 'F_g_XX'] = T_max_XX + 1


    # 8. Missing-treatment: conservative drop
    if drop_if_d_miss_before_first_switch:
        mask = (
            (df['min_time_d_miss_aft_ynm_XX'] < df['F_g_XX']) &
            (df['time_XX'] >= df['min_time_d_miss_aft_ynm_XX'])
        )
        df.loc[mask, 'outcome_XX'] = np.nan

    # 9. Missing-treatment: liberal conventions
    df['last_obs_D_bef_switch_t_XX'] = np.where(
        (df['time_XX'] < df['F_g_XX']) & df['treatment_XX'].notna(),
        df['time_XX'],
        np.nan
    )
    df['last_obs_D_bef_switch_XX'] = df.groupby('group_XX')['last_obs_D_bef_switch_t_XX'].transform('max')



    # drop outcomes before first non-missing D
    df.loc[df['time_XX'] < df['min_time_d_nonmiss_XX'], 'outcome_XX'] = np.nan

    # fill missing D before switch with baseline
    mask = (
        (df['F_g_XX'] < T_max_XX + 1) &
        df['treatment_XX'].isna() &
        (df['time_XX'] < df['last_obs_D_bef_switch_XX']) &
        (df['time_XX'] > df['min_time_d_nonmiss_XX'])
    )
    df.loc[mask, 'treatment_XX'] = df.loc[mask, 'd_sq_XX']



    # drop outcomes in ambiguous window and truncate controls there
    mask = (
        (df['F_g_XX'] < T_max_XX + 1) &
        (df['time_XX'] > df['last_obs_D_bef_switch_XX']) &
        (df['last_obs_D_bef_switch_XX'] < ( df['F_g_XX'] - 1 ))
    )
    df.loc[mask, 'outcome_XX'] = np.nan

    mask = (
        (df['F_g_XX'] < T_max_XX + 1) &
        (df['last_obs_D_bef_switch_XX'] < ( df['F_g_XX'] - 1 ))
    )
    df.loc[mask, 'trunc_control_XX'] = df.loc[mask, 'last_obs_D_bef_switch_XX'] + 1
    df.loc[mask, 'F_g_XX'] = T_max_XX + 1


    # carry forward post-switch D
    df['d_F_g_temp_XX'] = np.where(
        df['time_XX'] == df['F_g_XX'],
        df['treatment_XX'],
        np.nan
    )
    df['d_F_g_XX'] = df.groupby('group_XX')['d_F_g_temp_XX'].transform('mean')
    mask = (
        (df['F_g_XX'] < T_max_XX + 1) &
        df['treatment_XX'].isna() &
        (df['time_XX'] > df['F_g_XX']) &
        (df['last_obs_D_bef_switch_XX'] == df['F_g_XX'] - 1)
    )
    df.loc[mask, 'treatment_XX'] = df.loc[mask, 'd_F_g_XX']
    df = df.drop(columns=['d_F_g_temp_XX'])

    # for never-changers, fill mid-panel D and drop post-LD_g outcomes
    mask = (
        (df['F_g_XX'] == T_max_XX + 1) &
        df['treatment_XX'].isna() &
        (df['time_XX'] > df['min_time_d_nonmiss_XX']) &
        (df['time_XX'] < df['max_time_d_nonmiss_XX'])
    )
    df.loc[mask, 'treatment_XX'] = df.loc[mask, 'd_sq_XX']

    mask = (
        (df['F_g_XX'] == T_max_XX + 1) &
        (df['time_XX'] > df['max_time_d_nonmiss_XX'])
    )
    df.loc[mask, 'outcome_XX'] = np.nan
    df.loc[df['F_g_XX'] == T_max_XX + 1, 'trunc_control_XX'] = df['max_time_d_nonmiss_XX'] + 1

    # 10. Save outcome levels if predict_het
    if predict_het is not None and len(predict_het_good) > 0:
        df['outcome_non_diff_XX'] = df['outcome_XX']

    # 1. trends_lin adjustments
    if trends_lin:
        # Drop units with F_g_XX == 2
        df = df[df['F_g_XX'] != 2].copy()
        # Ensure sorted for group‐wise lag
        df = df.sort_values(['group_XX', 'time_XX'])
        # First‐differences for outcome and each control
        for v in ['outcome_XX'] + (controls or []):
            lag = df.groupby('group_XX')[v].shift(1)
            df[v] = df[v] - lag
        # Drop period 1 after differencing
        df = df[df['time_XX'] != 1].copy()
        t_min_XX = df['time_XX'].min()




    # 2. Balance the panel by filling missing (group_XX, time_XX) cells
    # Get full index
    # Drop any stray column
    df = df.drop(columns=[c for c in ['joint_trends_XX'] if c in df.columns])
    groups = df['group_XX'].unique()
    times = df['time_XX'].unique()
    full_index = pd.MultiIndex.from_product([groups, times], names=['group_XX','time_XX'])
    df = df.set_index(['group_XX','time_XX']).reindex(full_index).reset_index()


    # 3. Recompute numeric types
    df['group_XX'] = df['group_XX'].astype(int)
    df['time_XX']  = df['time_XX'].astype(int)

    # 4. Collapse baseline d_sq_XX by group
    df['d_sq_XX'] = df.groupby('group_XX')['d_sq_XX'].transform('mean')
    df["d_sq_int_XX"] = df.groupby("group_XX")["d_sq_int_XX"].transform(lambda x: x.mean(skipna=True))



    # 2. F_g_XX := mean(F_g_XX) by group_XX
    df["F_g_XX"] = df.groupby("group_XX")["F_g_XX"].transform(lambda x: x.mean(skipna=True))

    # 5. Define N_gt_XX
    df['N_gt_XX'] = np.where(
        df['outcome_XX'].isna() | df['treatment_XX'].isna(),
        0,
        df['weight_XX']
    )

    # 6. Compute F_g_trunc_XX
    # If F_g_XX < trunc_control_XX use F_g_XX else trunc_control_XX, 
    # but if trunc_control_XX is NA, use F_g_XX, 
    # and if F_g_XX is NA, use trunc_control_XX
    df['F_g_trunc_XX'] = np.where(
        df['F_g_XX'] < df['trunc_control_XX'],
        df['F_g_XX'],
        df['trunc_control_XX']
    )
    mask = df['trunc_control_XX'].isna()
    df.loc[mask, 'F_g_trunc_XX'] = df.loc[mask, 'F_g_XX']
    mask2 = df['F_g_XX'].isna()
    df.loc[mask2, 'F_g_trunc_XX'] = df.loc[mask2, 'trunc_control_XX']

    # 7. Compute T_g_XX by (d_sq_XX + trends_nonparam) groups
    if trends_nonparam:
        group_cols = ['d_sq_XX'] + trends_nonparam
    else:
        group_cols = ['d_sq_XX']

    df['T_g_XX'] = df.groupby(group_cols)['F_g_trunc_XX'].transform('max') - 1

    # 1. Compute average post‐switch treatment by group
    df['treatment_XX_v1'] = np.where(
        (df['time_XX'] >= df['F_g_XX']) & (df['time_XX'] <= df['T_g_XX']),
        df['treatment_XX'],
        np.nan
    )
    df['avg_post_switch_treat_XX_temp'] = (
        df.groupby('group_XX')['treatment_XX_v1']
        .transform('sum')
    )
    df.drop(columns='treatment_XX_v1', inplace=True)

    mask = ((df['time_XX'] >= df['F_g_XX']) &
        (df['time_XX'] <= df['T_g_XX']) )
    # 2. Count post‐switch periods by group
    df[ 'count_time_post_switch_XX_temp' ] = np.nan
    df[ 'count_time_post_switch_XX_temp'] =  df['treatment_XX'].notna().astype(int)
    df.loc[ ~mask, 'count_time_post_switch_XX_temp'] =  np.nan
        
    df['count_time_post_switch_XX'] = (
        df.groupby('group_XX')['count_time_post_switch_XX_temp']
        .transform('sum')
    )

    # 3. Finalize avg_post_switch_treat_XX
    df['avg_post_switch_treat_XX_temp'] = (
        df['avg_post_switch_treat_XX_temp'] / df['count_time_post_switch_XX']
    )
    df['avg_post_switch_treat_XX'] = (
        df.groupby('group_XX')['avg_post_switch_treat_XX_temp']
        .transform('mean')
    )
    df.drop(columns='avg_post_switch_treat_XX_temp', inplace=True)



    # 4. Define S_g_XX
    if continuous == 0:
        mask = (
            (df['avg_post_switch_treat_XX'] == df['d_sq_XX']) &
            (df['F_g_XX'] != (df['T_g_XX'] + 1)) 
        )
        df = df.loc[~mask]
        df['S_g_XX'] = (df['avg_post_switch_treat_XX'] > df['d_sq_XX']).astype(int)
        mask = df['avg_post_switch_treat_XX'].isnull()
        df.loc[ mask , 'S_g_XX'] = np.nan
        ### this is something we need to check 
        ### when we compare the missing with no values we need to check something
        mask = df['avg_post_switch_treat_XX'].isnull() & df['d_sq_XX'].notnull() & df[ 'S_g_XX'].isnull()
        df.loc[ mask , 'S_g_XX'] = 1
        mask = (df['F_g_XX'] != (T_max_XX + 1))
        df.loc[ ~mask , 'S_g_XX'] = np.nan 

    elif continuous > 0:
        mask = (
            (df['avg_post_switch_treat_XX'] == df['d_sq_XX_orig']) &
            df['avg_post_switch_treat_XX'].notna() &
            (df['F_g_XX'] != df['T_g_XX'] + 1) &
            df['F_g_XX'].notna() &
            df['T_g_XX'].notna()
        )
        df = df.loc[~mask]
        df['S_g_XX'] = (df['avg_post_switch_treat_XX'] > df['d_sq_XX_orig']).astype(int)
        df.loc[df['F_g_XX'] == T_max_XX + 1, 'S_g_XX'] = np.nan
    aux = df.groupby('group_XX')['avg_post_switch_treat_XX'].transform('min')
    df.loc[aux.isna(), 'S_g_XX'] = np.nan




    # 5. Define S_g_het_XX if needed
    if (predict_het and len(predict_het) > 0) or continuous > 0:
        df['S_g_het_XX'] = df['S_g_XX'].where(df['S_g_XX'] != 0, -1)


    if continuous > 0:
        # 6. Continuous‐specific binarization & stagger
        if controls is None:
            controls = []

        # treatment_XX_temp = (F_g_XX <= time_XX) * S_g_het_XX  if S_g_het_XX != .
        mask_het_notna = df["S_g_het_XX"].notna()
        df["treatment_XX_temp"] = np.where(
            mask_het_notna,
            (df["F_g_XX"] <= df["time_XX"]).astype(float) * df["S_g_het_XX"],
            np.nan
        )

        # treatment_XX_orig = treatment_XX
        df["treatment_XX_orig"] = df["treatment_XX"]

        # replace treatment_XX = treatment_XX_temp
        df["treatment_XX"] = df["treatment_XX_temp"]

        # ---- Create time_fe_XX_ dummies / step FEs ----
        # Stata: tab time_XX, gen(time_fe_XX_) and then replace time_fe_XX_i = (time_XX >= i)
        max_time_XX = int(df["time_XX"].max())

        for t in range(1, max_time_XX + 1):
            df[f"time_fe_XX_{t}"] = (df["time_XX"] >= t).astype(int)

        # ---- Interact period-step FEs with polynomial in d_sq_XX ----
        # foreach var of varlist time_fe_XX_2-time_fe_XX_`max_time_XX' {
        for t in range(2, max_time_XX + 1):
            var = f"time_fe_XX_{t}"
            for pol_level in range(1, degree_pol + 1):
                d_col = f"d_sq_{pol_level}_XX"
                newcol = f"{var}_bt{pol_level}_XX"
                df[newcol] = df[var] * df[d_col]
                controls.append(newcol)

            # capture drop `var'
            df.drop(columns=[var], inplace=True)

        # capture drop time_fe_XX_1
        if "time_fe_XX_1" in df.columns:
            df.drop(columns=["time_fe_XX_1"], inplace=True)






    # 1. Create treatment at F_g: D_{g,F_g}
    df['d_fg_XX'] = np.where(
        df['time_XX'] == df['F_g_XX'],
        df['treatment_XX'],
        np.nan
    )
    # group‐wise average
    df['d_fg_XX'] = df.groupby('group_XX')['d_fg_XX'].transform('mean')
    # if never switches (F_g == T_max + 1), fill from status‐quo d_sq_XX
    mask = df['d_fg_XX'].isna() & (df['F_g_XX'] == T_max_XX + 1)
    df.loc[ mask, 'd_fg_XX' ] = df.loc[ mask, 'd_sq_XX']

    # 2. Create L_g_XX = T_g_XX - F_g_XX + 1
    df['L_g_XX'] = df['T_g_XX'] - df['F_g_XX'] + 1




    # 3. Create L_g_placebo_XX if placebos requested
    if placebo > 0:
        df['L_g_placebo_XX'] = np.where(
            df['F_g_XX'] >= 3,
            np.where(
                df['L_g_XX'] > df['F_g_XX'] - 2,
                df['F_g_XX'] - 2,
                df['L_g_XX']
            ),
            np.nan
        )
        # replace infinite with NaN
        df.loc[np.isinf(df['L_g_placebo_XX']), 'L_g_placebo_XX'] = np.nan

    # 4. Tag first observation within each group_XX
    df = df.sort_values(['group_XX', 'time_XX'])
    df['first_obs_by_gp_XX'] = (df.groupby('group_XX').cumcount() == 0).astype(int)

    # 5. If clustering specified, flag first obs in each cluster and check nesting
    if cluster is not None:
        
        # 1. Generate cluster_group_XX = min(cluster) by group
        group_col = 'group_XX'
        cluster_col = 'cluster_XX'
        time_col = 'time_XX'
        df["cluster_group_XX"] = df.groupby(group_col)[cluster_col].transform("min")

        # Replace missing cluster with cluster_group_XX
        df[cluster_col] = np.where(df[cluster_col].isna(), df["cluster_group_XX"], df[cluster_col])

        # 2. First observation by cluster (sorted within group and time)
        df = df.sort_values([cluster_col, group_col, time_col])
        df["first_obs_by_clust_XX"] = (
            df.groupby(cluster_col).cumcount().eq(0).astype(int)
        )
        df.loc[df[cluster_col].isna(), "first_obs_by_clust_XX"] = np.nan

        # 3. Error check: group must be nested in cluster
        # If within a group, cluster has >1 unique value → not nested
        cluster_var = (
            df.groupby(group_col)[cluster_col]
            .nunique(dropna=True)
            .reset_index(name="cluster_var_g_XX")
        )
        max_cluster_var = cluster_var["cluster_var_g_XX"].max()

        if max_cluster_var > 1:
            raise ValueError(
                "❌ The group variable should be nested within the clustering variable."
            )

    # 6. Compute first differences of outcome and treatment
    df = df.sort_values(['group_XX', 'time_XX'])  # ensure sorted like xtset
    df['diff_y_XX'] = df.groupby('group_XX')['outcome_XX'].diff()
    df['diff_d_XX'] = df.groupby('group_XX')['treatment_XX'].diff()


    if controls is not None:

        # 1) Compute first differences of each control and flag missing
        count_controls = 0
        df['fd_X_all_non_missing_XX'] = 1

        for var in controls:
            count_controls += 1
            diff_col = f'diff_X{count_controls}_XX'
            # group‐wise first difference
            df[diff_col] = df.groupby('group_XX')[var].diff()
            # if diff is NaN, mark as missing
            df['fd_X_all_non_missing_XX'] = np.where(
                df[diff_col].isna(), 
                0, 
                df['fd_X_all_non_missing_XX']
            )

        # 2) Residualization prep
        count_controls = 0
        mycontrols_XX = []

        for var in controls:
            count_controls += 1
            diff_col = f'diff_X{count_controls}_XX'

            # remove any stale helpers
            for tmp in ['sum_weights_control_XX','avg_diff_temp_XX','diff_y_wXX']:
                if tmp in df.columns:
                    df.drop(columns=tmp, inplace=True)

            # define mask: not-yet-switched & valid diff_y
            mask = (
                (df['ever_change_d_XX'] == 0) &
                df['diff_y_XX'].notna() &
                (df['fd_X_all_non_missing_XX'] == 1)
            )

            # grouping keys
            grp_cols = ['time_XX','d_sq_XX'] + (trends_nonparam or [])

            # 2a) sum of N_gt for controls
            df['sum_weights_control_XX'] = (
                df.loc[mask]
                .groupby(grp_cols)['N_gt_XX']
                .transform('sum')
            )
            df.loc[~mask, 'sum_weights_control_XX'] = np.nan

            # 2b) weighted sum of first-diffs
            df['avg_diff_temp_XX'] = df['N_gt_XX'] * df[diff_col]
            avg_col = f'avg_diff_X{count_controls}_XX'
            df[avg_col] = (
                df.loc[mask]
                .groupby(grp_cols)['avg_diff_temp_XX']
                .transform('sum')
            )
            df.loc[~mask, avg_col] = np.nan
            df[avg_col] = df[avg_col] / df['sum_weights_control_XX']

            # 2c) residual (√N * (ΔX - avg ΔX))
            resid_col = f'resid_X{count_controls}_time_FE_XX'
            df[resid_col] = np.sqrt(df['N_gt_XX']) * (df[diff_col] - df[avg_col])
            df[resid_col] = df[resid_col].fillna(0)

            mycontrols_XX.append(resid_col)

            # 2d) prepare product with ΔY
            df['diff_y_wXX'] = np.sqrt(df['N_gt_XX']) * df['diff_y_XX']
            prod_col = f'prod_X{count_controls}_Ngt_XX'
            df[prod_col] = df[resid_col] * np.sqrt(df['N_gt_XX'])
            df[prod_col] = df[prod_col].fillna(0)

        # Prepare storage
        store_singular = {}               # flags by index
        store_noresidualization_XX = []   # list of levels dropped
        levels_d_sq_XX_final = []         # levels for which we computed coefs
        # Get unique levels of d_sq_int_XX
        levels_d_sq_XX = df['d_sq_int_XX'].astype('category').cat.categories

        # Dictionaries to hold results
        coefs_sq = {}
        inv_Denom = {}

        # Loop over each baseline‐treatment level
        for idx, l in enumerate(levels_d_sq_XX, start=1):
            l = int(l)
            # Count distinct F_g_XX for this level
            dict_glob[f"useful_res_{int(l)}_XX"] = df.loc[df['d_sq_int_XX'] == l, 'F_g_XX'].nunique()
            store_singular[idx] = False

            if dict_glob[f"useful_res_{int(l)}_XX"] > 1:
                # Subset to the observations used for theta_d
                mask = (
                    (df['ever_change_d_XX'] == 0) &
                    df['diff_y_XX'].notna() &
                    (df['fd_X_all_non_missing_XX'] == 1) &
                    (df['d_sq_int_XX'] == l)
                )
                data_XX = df.loc[mask, :]

                # Build YX matrix: [Y, X_residuals..., 1]
                Y_vec = data_XX['diff_y_wXX'].to_numpy()
                X_vec = data_XX[mycontrols_XX].to_numpy()
                ones = np.ones((len(Y_vec), 1))
                YX = np.hstack([Y_vec.reshape(-1,1), X_vec, ones])

                # Compute cross‐product matrix
                overall = YX.T @ YX

                # Check if entire matrix is NaN
                e_vec = np.ones((1, overall.shape[1]))
                val = e_vec @ overall @ e_vec.T
                if np.isnan(val)[0,0]:
                    # Singular: cannot invert or accumulate
                    store_singular[idx] = True
                    store_noresidualization_XX.append(l)
                    dict_glob[f"useful_res_{int(l)}_XX"] = 1
                else:
                    # Extract the (k × k) block for controls
                    k = len(mycontrols_XX)
                    M = overall[1:1+k, 1:1+k]
                    v = overall[1:1+k, 0]

                    # Compute θ_d via Moore-Penrose inverse
                    theta_d = np.linalg.pinv(M) @ v
                    dict_glob[f'coefs_sq_{l}_XX'] = theta_d
                    levels_d_sq_XX_final.append(l)
                    # Check invertibility
                    if abs(np.linalg.det(M)) <= 1e-16:
                        store_singular[idx] = True
                    rmax = df['F_g_XX'].max()
                    df[f"N_c_{l}_temp_XX"] = np.nan
                    df[f"N_c_{l}_temp_XX"] = ((df["time_XX"] >= 2) & (df["time_XX"] <= (rmax - 1)) & (df['time_XX'] < df['F_g_XX']) & (df['diff_y_XX'].notnull()))
                    rsum = df.loc[ df[f"N_c_{l}_temp_XX"] == 1, 'N_gt_XX'].sum()

                    # Store inverse Denominator scaled by G_XX
                    dict_glob[f"inv_Denom_{l}_XX"] = np.linalg.pinv(M) * rsum * G_XX
        
        # Reconstruct store_singular_XX string using original d_sq_XX levels
        store_singular_XX = ""
        levels_d_sq_bis_XX = df['d_sq_XX'].astype('category').cat.categories
        for idx, l in enumerate(levels_d_sq_bis_XX, start=1):
            l = int(l)
            idx = int(idx)
            if store_singular.get(idx, False):
                store_singular_XX += f" {l}"


        # 1. Display warnings if any Den_d^{-1} was singular
        if store_singular_XX.strip():
            warnings.warn(
                "Some control variables are not taken into account for groups with baseline "
                f"treatment equal to:{store_singular_XX}"
            )
            warnings.warn(
                "1. For these groups, the regression of ΔY on ΔX and time‐FE had fewer "
                "observations than regressors."
            )
            warnings.warn(
                "2. For these groups, one or more controls were perfectly collinear (no time variation)."
            )

        # 2. Drop levels where residualization failed entirely
        if store_noresidualization_XX:
            df = df[~df["d_sq_int_XX"].isin(store_noresidualization_XX)].copy()

        # 3. Prepare for the “fixed‐effect” residualization regressions
        #    Create a categorical time FE (we’ll re‐level to reference=2 below)
        df["time_FE_XX"] = df["time_XX"].astype("category")

        # 4. Loop over each baseline‐treatment level we actually residualized
        for l in levels_d_sq_XX_final:
            l = int(l)
            outcol = f"E_y_hat_gt_int_{l}_XX"
            df[outcol] = 0.0

            # subset of rows used in that regression
            mask = (
                (df["d_sq_int_XX"] == l)
                & (df["F_g_XX"] > df["time_XX"])
            )
            data_reg = df.loc[mask].copy()

            # reorder the categories so that 2 is the base level
            cats = list(data_reg["time_FE_XX"].cat.categories)
            if 2 in cats:
                new_order = [2] + [c for c in cats if c != 2]
                data_reg["time_FE_XX"] = data_reg["time_FE_XX"].cat.reorder_categories(
                    new_order, ordered=True
                )

            # build the formula: diff_y_XX ~ diff_X1_XX + ... + diff_Xk_XX + C(time_FE_XX) -1
            fe_terms = [f"diff_X{c}_XX" for c in range(1, count_controls + 1)]
            formula = "diff_y_XX ~ " + " + ".join(fe_terms + ["C(time_FE_XX)"]) + " -1"

            # fit weighted least squares
            model = smf.wls(formula, data=data_reg, weights=data_reg["weight_XX"]).fit()
            data_reg["y_hat"] = model.predict(data_reg)
            df[ outcol ] = np.nan
            df.loc[ mask, outcol] = data_reg["y_hat"].tolist()


            # # add up each coefficient times its column in the full df
            # for varname, coef in model.params.items():
            #     df["__tmp"] = df.get(varname, 0) * coef
            #     df["__tmp"] = df["__tmp"].fillna(0)
            #     df[outcol] += df["__tmp"]
            # df.loc[~mask, outcol] = np.nan
            # df.drop(columns="__tmp", inplace=True)

        # 5. Clean up any numeric dummy columns if you created them earlier:
        for t in range(2, int(T_max_XX) + 1):
            t = int(t)
            col = f"time_FE_XX{t}"
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # 6. Drop the temporary factor column
        df.drop(columns="time_FE_XX", inplace=True)

        
    # Initialize
    L_u_XX = None
    L_a_XX = None
    L_placebo_u_XX = None
    L_placebo_a_XX = None

    # For switchers in
    if switchers in ("", "in"):
        cond_in = df["S_g_XX"] == 1
        if cond_in.any():
            L_u_XX = df.loc[cond_in, "L_g_XX"].max()
        else:
            L_u_XX = 0

        if placebo > 0:
            cond_pl_in = df["S_g_XX"] == 1
            if cond_pl_in.any() and "L_g_placebo_XX" in df:
                L_placebo_u_XX = df.loc[cond_pl_in, "L_g_placebo_XX"].max()
            else:
                L_placebo_u_XX = 0
            # Enforce non-negativity
            L_placebo_u_XX = max(L_placebo_u_XX, 0)
            if trends_lin:
                L_placebo_u_XX -= 1

    if switchers in ("", "out"):
        # compute L_a_XX
        mask = df["S_g_XX"] == 0
        vals = df.loc[mask, "L_g_XX"]

        if vals.dropna().empty:
            L_a_XX = 0
        else:
            L_a_XX = np.nanmax(vals)

        # placebo case
        if placebo != 0:
            vals_placebo = df.loc[mask, "L_g_placebo_XX"]

            if vals_placebo.dropna().empty:
                L_placebo_a_XX = 0
            else:
                L_placebo_a_XX = np.nanmax(vals_placebo)

            # replace negatives with 0
            if L_placebo_a_XX < 0:
                L_placebo_a_XX = 0

            # subtract 1 if trends_lin is True
            if bool(trends_lin):
                L_placebo_a_XX = L_placebo_a_XX - 1


    # Design‐restriction check
    if (
        (switchers == "in" and (L_u_XX is None or L_u_XX == 0)) or
        (switchers == "out" and (L_a_XX is None or L_a_XX == 0)) or
        (switchers == "" and ((L_u_XX is None or L_u_XX == 0) and (L_a_XX is None or L_a_XX == 0)))
    ):
        raise ValueError(
            "No treatment effect can be estimated.\n"
            "This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) "
            "is not satisfied in the data, given the options requested. "
            "This may be due to continuous-period-one treatments or lack of variation. "
            "Try specifying the continuous option or check your data."
        )

    # Determine the number of effects to estimate
    if switchers == "":
        l_XX = int(min(np.nanmax(np.array([L_a_XX, L_u_XX])), effects))
        if placebo != 0:
            l_placebo_XX = np.nanmax(np.array([L_placebo_a_XX, L_placebo_u_XX]))
            l_placebo_XX = np.nanmin( np.array( [ l_placebo_XX, placebo ] ) )
            l_placebo_XX = np.nanmin( np.array( [ l_placebo_XX, effects ] ) )
        else:
            l_placebo_XX = 0

    elif switchers == "in":
        l_XX = int(np.nanmin(np.array([effects, L_u_XX])))
        if placebo != 0:
            l_placebo_XX = int( np.nanmin( np.array( [ placebo, L_placebo_u_XX ] ) ) )
            l_placebo_XX = int( np.nanmin( np.array( [ l_placebo_XX, effects ] ) ) )
        else:
            l_placebo_XX = 0

    else:  # switchers == "out"
        l_XX = int(np.nanmin(np.array([effects, L_a_XX])))
        if placebo != 0:
            l_placebo_XX = int( np.nanmin( np.array( [ placebo, L_placebo_a_XX ] ) ) )
            l_placebo_XX = int( np.nanmin( np.array( [ l_placebo_XX, effects ] ) ) )
        else:
            l_placebo_XX = 0

    # 1. Warn if the user requested too many effects or placebos
    if l_XX < effects:
        warnings.warn(
            f"The number of effects requested is too large. "
            f"The number of effects which can be estimated is at most {l_XX}. "
            f"Trying to estimate {l_XX} effect(s)."
        )

    if placebo != 0:
        if l_placebo_XX < placebo and effects >= placebo:
            warnings.warn(
                f"The number of placebos which can be estimated is at most {l_placebo_XX}. "
                f"Trying to estimate {l_placebo_XX} placebo(s)."
            )
        if effects < placebo:
            warnings.warn(
                f"The number of placebos requested cannot be larger than the number "
                f"of effects requested. Cannot compute more than {l_placebo_XX} placebo(s)."
            )

    # 2. Compute adjustment windows for placebo tests
    df['pl_gap_XX'] = np.where(
        df['S_g_XX'].notna(),
        df['F_g_XX'] - 2 - df['L_g_XX'],
        np.nan
    )

    max_pl_u_XX = max_pl_a_XX = max_pl_gap_u_XX = max_pl_gap_a_XX = 0
    if switchers in ("", "in"):
        if (df['S_g_XX'] == 1).any():
            max_pl_u_XX = df.loc[df['S_g_XX'] == 1, 'F_g_XX'].max() - 2
            max_pl_gap_u_XX = df.loc[df['S_g_XX'] == 1, 'pl_gap_XX'].max()
    if switchers in ("", "out"):
        if (df['S_g_XX'] == 0).any():
            max_pl_a_XX = df.loc[df['S_g_XX'] == 0, 'F_g_XX'].max() - 2
            max_pl_gap_a_XX = df.loc[df['S_g_XX'] == 0, 'pl_gap_XX'].max()

    max_pl_XX      = max(max_pl_u_XX,      max_pl_a_XX)
    max_pl_gap_XX  = max(max_pl_gap_u_XX,  max_pl_gap_a_XX)

    # clean up temporary column
    df.drop(columns='pl_gap_XX', inplace=True)

    # 3. Initialize accumulation variables and DataFrame columns
    inh_obj = []

    # a) For each effect k = 1..l_XX, create zeroed columns
    for k in range(1, l_XX + 1):
        k = int(k)
        df[f"U_Gg{k}_plus_XX"]      = 0
        df[f"U_Gg{k}_minus_XX"]     = 0
        df[f"count{k}_plus_XX"]     = 0
        df[f"count{k}_minus_XX"]    = 0
        df[f"U_Gg_var_{k}_in_XX"]   = 0
        df[f"U_Gg_var_{k}_out_XX"]  = 0
        df[f"delta_D_g_{k}_plus_XX"]= 0
        df[f"delta_D_g_{k}_minus_XX"]= 0

    # b) Global counters for in-group and out-group variance sums
    sum_for_var_in_XX  = 0
    sum_for_var_out_XX = 0
    inh_obj.extend(['sum_for_var_in_XX', 'sum_for_var_out_XX'])

    # c) If placebo tests requested, initialize their columns too
    if placebo != 0:
        for k in range(1, l_XX + 1):
            k=int(k)
            df[f"U_Gg_pl_{k}_plus_XX"]     = 0
            df[f"U_Gg_pl_{k}_minus_XX"]    = 0
            df[f"count{k}_pl_plus_XX"]     = 0
            df[f"count{k}_pl_minus_XX"]    = 0
            df[f"U_Gg_var_pl_{k}_in_XX"]   = 0
            df[f"U_Gg_var_pl_{k}_out_XX"]  = 0
        sum_for_var_placebo_in_XX  = 0
        sum_for_var_placebo_out_XX = 0
        inh_obj.extend(['sum_for_var_placebo_in_XX', 'sum_for_var_placebo_out_XX'])

    # d) Create de-weighted and raw counters per effect / placebo
    for i in range(1, l_XX + 1):
        i=int(i)
        # de-weighted and raw cell counts
        dict_glob[f"N1_{i}_XX"]      = 0
        dict_glob[f"N1_{i}_XX_new"]  = 0
        dict_glob[f"N1_dw_{i}_XX"]   = 0
        dict_glob[f"N0_{i}_XX"]      = 0
        dict_glob[f"N0_{i}_XX_new"]  = 0
        dict_glob[f"N0_dw_{i}_XX"]   = 0

        inh_obj.extend([
            f"N1_{i}_XX", f"N1_{i}_XX_new", f"N1_dw_{i}_XX",
            f"N0_{i}_XX", f"N0_{i}_XX_new", f"N0_dw_{i}_XX",
        ])

        # optionally track normalized deltas
        if normalized:
            dict_glob[f"delta_D_{i}_in_XX"]  = 0
            dict_glob[f"delta_D_{i}_out_XX"] = 0
            inh_obj.extend([f"delta_D_{i}_in_XX", f"delta_D_{i}_out_XX"])

        # placebo-specific counters
        if placebo != 0:
            dict_glob[f"N1_placebo_{i}_XX"]      = 0
            dict_glob[f"N1_placebo_{i}_XX_new"]  = 0
            dict_glob[f"N1_dw_placebo_{i}_XX"]   = 0
            dict_glob[f"N0_placebo_{i}_XX"]      = 0
            dict_glob[f"N0_placebo_{i}_XX_new"]  = 0
            dict_glob[f"N0_dw_placebo_{i}_XX"]   = 0
            inh_obj.extend([
                f"N1_placebo_{i}_XX", f"N1_placebo_{i}_XX_new", f"N1_dw_placebo_{i}_XX",
                f"N0_placebo_{i}_XX", f"N0_placebo_{i}_XX_new", f"N0_dw_placebo_{i}_XX",
            ])
            if normalized:
                dict_glob[f"delta_D_pl_{i}_in_XX"]  = 0
                dict_glob[f"delta_D_pl_{i}_out_XX"] = 0
                inh_obj.extend([f"delta_D_pl_{i}_in_XX", f"delta_D_pl_{i}_out_XX"])

    # 1. Initialize DataFrame columns and scalar counters
    df['U_Gg_plus_XX'] = 0
    df['U_Gg_minus_XX'] = 0

    U_Gg_den_plus_XX = 0
    U_Gg_den_minus_XX = 0
    sum_N1_l_XX       = 0
    sum_N0_l_XX       = 0
    inh_obj.extend([
        "U_Gg_den_plus_XX",
        "U_Gg_den_minus_XX",
        "sum_N1_l_XX",
        "sum_N0_l_XX"
    ])

    df['U_Gg_var_plus_XX']  = 0
    df['U_Gg_var_minus_XX'] = 0

    # 2. Collect inherited scalars into a dict
    const = {}
    for v in inh_obj:
        if v in dict_glob.keys():
            const[v] = dict_glob.get(v)
        elif v in locals().keys():
            const[v] = locals().get(v)
        else:
            const[v] = np.nan


    # 3. Save unchanging globals
    gs = ["L_u_XX", "L_a_XX", "l_XX", "t_min_XX", "T_max_XX", "G_XX"]
    if placebo != 0:
        gs += ["L_placebo_u_XX", "L_placebo_a_XX"]


    globals_dict = {}
    for v in gs:
        if v in dict_glob.keys():
            globals_dict[v] = dict_glob.get(v)
        elif v in locals().keys():
            globals_dict[v] = locals().get(v)
        else:
            globals_dict[v] = np.nan

    # 4. Collect control-specific objects if any
    controls_globals = {}
    if controls is not None:
        for lvl in levels_d_sq_XX:
            controls_globals[f"useful_res_{int(lvl)}_XX"] = dict_glob.get(f"useful_res_{int(lvl)}_XX")
            controls_globals[f"coefs_sq_{int(lvl)}_XX"]     = dict_glob.get(f"coefs_sq_{int(lvl)}_XX")
            controls_globals[f"inv_Denom_{int(lvl)}_XX"]     = dict_glob.get(f"inv_Denom_{int(lvl)}_XX")

    # 5. Tag switchers by event-study effect number
    df['switchers_tag_XX'] = np.nan


    # df_const_py = pd.DataFrame([const])
    # df_const_py = df_const_py.transpose().reset_index().copy()
    # df_const_py.columns = [ "key_info",  "value" ]
    # df_const_py.head()


    # # Rscript return of const
    # df_const_inR = pd.read_csv('cons_main_line971.csv')
    # df_const_inR

    # df_merge = df_const_inR.merge( df_const_py, on = 'key_info' , how = 'outer', indicator = True )
    # df_merge._merge.value_counts()



    # df: pandas DataFrame
    # const: dict to store dynamic values
    # did_multiplegt_dyn_core: your core estimation function
    # print(l_XX, globals_dict['T_max_XX'])
    # -------------------------------
    # Switchers "in"
    # -------------------------------
    dict_glob['U_Gg_den_plus_XX'] = 0
    dict_glob['U_Gg_den_minus_XX'] = 0
    dict_glob['sum_N1_l_XX'] = 0
    dict_glob['sum_N0_l_XX'] = 0

    if switchers in ["", "in"]:
        if L_u_XX is not None and L_u_XX != 0:

            if not trends_lin: # if trends lin is false
                
                data = did_multiplegt_dyn_core(
                    df,
                    outcome="outcome_XX",
                    group="group_XX",
                    time="time_XX",
                    treatment="treatment_XX",
                    effects=l_XX,
                    placebo=l_placebo_XX,
                    switchers_core="in",
                    trends_nonparam=trends_nonparam,
                    controls=controls,
                    same_switchers=same_switchers,
                    same_switchers_pl=same_switchers_pl,
                    only_never_switchers=only_never_switchers,
                    normalized=normalized,
                    globals_dict=globals_dict,
                    dict_glob=dict_glob,
                    const=const,
                    trends_lin=trends_lin,
                    controls_globals=controls_globals,
                    less_conservative_se=less_conservative_se,
                    continuous=continuous,
                    cluster=cluster,
                    **const
                )
                
                
                df = data["df"]
                for keyval in list(data["const"].keys()):
                    const[f"{keyval}"] = data["const"][f"{keyval}"]
                    dict_glob[f"{keyval}"] = data["const"][f"{keyval}"]

                for k in range(1, l_XX + 1):
                    k=int(k)
                    mask = df[f"distance_to_switch_{k}_XX"] == 1
                    df.loc[mask, "switchers_tag_XX"] = k

                
            for i in range(1, l_XX + 1):
                i=int(i)
                if trends_lin:
                    data = did_multiplegt_dyn_core(
                        df,
                        outcome="outcome_XX",
                        group="group_XX",
                        time="time_XX",
                        treatment="treatment_XX",
                        effects=i,
                        placebo=0,
                        switchers_core="in",
                        trends_nonparam=trends_nonparam,
                        controls=controls,
                        same_switchers=same_switchers, #change 1
                        same_switchers_pl=same_switchers_pl,
                        only_never_switchers=only_never_switchers,
                        normalized=normalized,
                        globals_dict=globals_dict,
                        dict_glob=dict_glob,
                        const=const,
                        trends_lin=trends_lin,
                        controls_globals=controls_globals,
                        less_conservative_se=less_conservative_se,
                        continuous=continuous,
                        cluster=cluster,
                        **const
                    )
                    df = data["df"]
                    for keyval in list(data["const"].keys()):
                        const[f"{keyval}"] = data["const"][f"{keyval}"]
                        dict_glob[f"{keyval}"] = data["const"][f"{keyval}"]

                    mask = df[f"distance_to_switch_{i}_XX"] == 1
                    df.loc[mask, "switchers_tag_XX"] = i

                if f"N1_{i}_XX" in list(dict_glob.keys()):
                    N1_i = dict_glob.get(f"N1_{i}_XX")
                    print(f"{N1_i} N1_{i}_XX")
                else:
                    print(f"Warning: N1_{i}_XX not found in dict_glob keys.")
                    N1_i = "hola"

                if N1_i != 0:
                    df[f"U_Gg{i}_plus_XX"]    = df[f"U_Gg{i}_XX"]
                    df[f"count{i}_plus_XX"]   = df[f"count{i}_core_XX"]
                    df[f"U_Gg_var_{i}_in_XX"] = df[f"U_Gg{i}_var_XX"]

                    dict_glob[f"N1_{i}_XX_new"] = N1_i
                    const[f"N1_{i}_XX_new"] = N1_i

                    if normalized:
                        dict_glob[f"delta_D_{i}_in_XX"] = dict_glob.get(f"delta_norm_{i}_XX")
                        const[f"delta_D_{i}_in_XX"] = dict_glob[f"delta_D_{i}_in_XX"]

                    if not trends_lin:
                        df[f"delta_D_g_{i}_plus_XX"] = df[f"delta_D_g_{i}_XX"]

            if l_placebo_XX != 0:
                for i in range(1, int(l_placebo_XX + 1)):
                    i=int(i)
                    if trends_lin:
                        merged = const | controls_globals
                        data = did_multiplegt_dyn_core(
                            df,
                            outcome="outcome_XX",
                            group="group_XX",
                            time="time_XX",
                            treatment="treatment_XX",
                            effects=i,
                            placebo=i,
                            switchers_core="in",
                            trends_nonparam=trends_nonparam,
                            controls=controls,
                            same_switchers=same_switchers,
                            same_switchers_pl=same_switchers_pl,
                            only_never_switchers=only_never_switchers,
                            normalized=normalized,
                            globals_dict=globals_dict,
                            dict_glob=dict_glob,
                            const=const,
                            trends_lin=trends_lin,
                            controls_globals=controls_globals,
                            less_conservative_se=less_conservative_se,
                            continuous=continuous,
                            cluster=cluster,
                            **const
                        )
                        df = data["df"]
                        for keyval in list(data["const"].keys()):
                            const[f"{keyval}"] = data["const"][f"{keyval}"]
                            dict_glob[f"{keyval}"] = data["const"][f"{keyval}"]

                        mask = df[f"distance_to_switch_{i}_XX"] == 1
                        df.loc[mask, "switchers_tag_XX"] = i

                    N1_pl = dict_glob.get(f"N1_placebo_{i}_XX", 0)
                    if N1_pl != 0:
                        df[f"U_Gg_pl_{i}_plus_XX"]   = df[f"U_Gg_placebo_{i}_XX"]
                        df[f"count{i}_pl_plus_XX"]   = df[f"count{i}_pl_core_XX"]
                        df[f"U_Gg_var_pl_{i}_in_XX"] = df[f"U_Gg_pl_{i}_var_XX"]

                        dict_glob[f"N1_placebo_{i}_XX_new"] = N1_pl
                        const[f"N1_placebo_{i}_XX_new"] = N1_pl

                        if normalized:
                            dict_glob[f"delta_D_pl_{i}_in_XX"] = dict_glob.get(f"delta_norm_pl_{i}_XX", 0)
                            const[f"delta_D_pl_{i}_in_XX"] = dict_glob.get(f"delta_D_pl_{i}_in_XX", 0)

            
            if not trends_lin and dict_glob['sum_N1_l_XX'] != 0:
                df["U_Gg_plus_XX"] = df["U_Gg_XX"]
                df["U_Gg_den_plus_XX"] = df["U_Gg_den_XX"]
                df["U_Gg_var_plus_XX"] = df["U_Gg_var_XX"]
    
    

    if switchers in ["", "out"]:
        if bool(~np.isnan(L_a_XX)) and L_a_XX != 0:

            if not trends_lin:
                data = did_multiplegt_dyn_core(
                    df,
                    outcome="outcome_XX",
                    group="group_XX",
                    time="time_XX",
                    treatment="treatment_XX",
                    effects=l_XX,
                    placebo=l_placebo_XX,
                    switchers_core="out",
                    trends_nonparam=trends_nonparam,
                    controls=controls,
                    same_switchers=same_switchers,
                    same_switchers_pl=same_switchers_pl,
                    only_never_switchers=only_never_switchers,
                    normalized=normalized,
                    globals_dict=globals_dict,
                    dict_glob=dict_glob,
                    const=const,
                    trends_lin=trends_lin,
                    controls_globals=controls_globals,
                    less_conservative_se=less_conservative_se,
                    continuous=continuous,
                    cluster=cluster,
                    **const
                )

                df = data["df"]
                for e, val in data["const"].items():
                    const[e] = val
                    dict_glob[e] = val

                for k in range(1, l_XX + 1):
                    k=int(k)
                    mask = df[f"distance_to_switch_{k}_XX"] == 1
                    df.loc[mask, "switchers_tag_XX"] = k

            for i in range(1, l_XX + 1):
                i=int(i)
                if trends_lin:
                    merged = const | controls_globals
                    data = did_multiplegt_dyn_core(
                        df,
                        outcome="outcome_XX",
                        group="group_XX",
                        time="time_XX",
                        treatment="treatment_XX",
                        effects=i,
                        placebo=0,
                        switchers_core="out",
                        trends_nonparam=trends_nonparam,
                        controls=controls,
                        same_switchers=same_switchers,
                        same_switchers_pl=same_switchers_pl,
                        only_never_switchers=only_never_switchers,
                        normalized=normalized,
                        globals_dict=globals_dict,
                        dict_glob=dict_glob,
                        const=const,
                        trends_lin=trends_lin,
                        controls_globals=controls_globals,
                        less_conservative_se=less_conservative_se,
                        continuous=continuous,
                        cluster=cluster,
                        **const
                    )
                    df = data["df"]
                    for keyval in list(data["const"].keys()):
                        const[f"{keyval}"] = data["const"][f"{keyval}"]
                        dict_glob[f"{keyval}"] = data["const"][f"{keyval}"]

                    mask = df[f"distance_to_switch_{i}_XX"] == 1
                    df.loc[mask, "switchers_tag_XX"] = i

                if f"N0_{i}_XX" in list(dict_glob.keys()):
                    N0_i = dict_glob.get(f"N0_{i}_XX")
                else:
                    print(f"Warning: N0_{i}_XX not found in dict_glob keys.")
                    N0_i = "hola"

                if N0_i != 0:
                    df[f"U_Gg{i}_minus_XX"] = -df[f"U_Gg{i}_XX"].copy()
                    df[f"count{i}_minus_XX"] = df[f"count{i}_core_XX"].copy()
                    df[f"U_Gg_var_{i}_out_XX"] = -df[f"U_Gg{i}_var_XX"].copy()

                    dict_glob[f"N0_{i}_XX_new"] = N0_i
                    const[f"N0_{i}_XX_new"] = N0_i

                    if normalized:
                        dict_glob[f"delta_D_{i}_out_XX"] = dict_glob.get(f"delta_norm_{i}_XX")
                        const[f"delta_D_{i}_out_XX"] = dict_glob[f"delta_D_{i}_out_XX"]

                    if not trends_lin:
                        df[f"delta_D_g_{i}_minus_XX"] = df[f"delta_D_g_{i}_XX"]

            if l_placebo_XX != 0:
                for i in range(1, int(int(l_placebo_XX) + 1)):
                    i=int(i)
                    if trends_lin:
                        merged = const | controls_globals
                        data = did_multiplegt_dyn_core(
                            df,
                            outcome="outcome_XX",
                            group="group_XX",
                            time="time_XX",
                            treatment="treatment_XX",
                            effects=i,
                            placebo=i,
                            switchers_core="out",
                            trends_nonparam=trends_nonparam,
                            controls=controls,
                            same_switchers=same_switchers,
                            same_switchers_pl=same_switchers_pl,
                            only_never_switchers=only_never_switchers,
                            normalized=normalized,
                            globals_dict=globals_dict,
                            dict_glob=dict_glob,
                            const=const,
                            trends_lin=trends_lin,
                            controls_globals=controls_globals,
                            less_conservative_se=less_conservative_se,
                            continuous=continuous,
                            cluster=cluster,
                            merged = const | controls_globals
                        )
                        df = data["df"]
                        for keyval in list(data["const"].keys()):
                            const[f"{keyval}"] = data["const"][f"{keyval}"]
                            dict_glob[f"{keyval}"] = data["const"][f"{keyval}"]

                        mask = df[f"distance_to_switch_{k}_XX"] == 1
                        df.loc[mask, "switchers_tag_XX"] = k

                    if f"N0_placebo_{i}_XX" in list(dict_glob.keys()):
                        N0_pl = dict_glob.get(f"N0_placebo_{i}_XX")
                    else:
                        print(f"Warning: N0_placebo_{i}_XX not found in dict_glob keys.")
                        N0_pl = "hola"
                        
                    N0_pl = dict_glob.get(f"N0_placebo_{i}_XX", 0)
                    if N0_pl != 0:
                        df[f"U_Gg_pl_{i}_minus_XX"] = -df[f"U_Gg_placebo_{i}_XX"]
                        df[f"count{i}_pl_minus_XX"] = df[f"count{i}_pl_core_XX"]
                        df[f"U_Gg_var_pl_{i}_out_XX"] = -df[f"U_Gg_pl_{i}_var_XX"]

                        dict_glob[f"N0_placebo_{i}_XX_new"] = N0_pl
                        const[f"N0_placebo_{i}_XX_new"] = N0_pl

                        if normalized:
                            dict_glob[f"delta_D_pl_{i}_out_XX"] = dict_glob.get(f"delta_norm_pl_{i}_XX", 0)
                            const[f"delta_D_pl_{i}_out_XX"] = dict_glob.get(f"delta_D_pl_{i}_out_XX", 0)

            if not trends_lin and dict_glob['sum_N0_l_XX'] != 0:
                df["U_Gg_minus_XX"] = -df["U_Gg_XX"]
                df["U_Gg_den_minus_XX"] = df["U_Gg_den_XX"]
                df["U_Gg_var_minus_XX"] = -df["U_Gg_var_XX"]

    # Collect rownames (placeholder)
    rownames = []

    # Initialize result matrix
    mat_res_XX = np.full((int(l_XX + l_placebo_XX +1 ), 9), np.nan)

    # --------------------------
    # Loop over effects
    # --------------------------
    for i in range(1, l_XX + 1):
        i=int(i)
        N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
        N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
        print("Number of N1")
        print(N1_new)
        print("Number of N0")
        print(N0_new)
        # U_Gg global
        df[f"U_Gg{i}_global_XX"] = (
            (N1_new / (N1_new + N0_new)) * df[f"U_Gg{i}_plus_XX"]
            + (N0_new / (N1_new + N0_new)) * df[f"U_Gg{i}_minus_XX"]
        )
        df.loc[df["first_obs_by_gp_XX"] == 0, f"U_Gg{i}_global_XX"] = np.nan

        # assuming i is already defined
        col_plus = f"count{i}_plus_XX"
        col_minus = f"count{i}_minus_XX"
        col_global = f"count{i}_global_XX"

        # Step 1: max between plus and minus
        df[col_global] = np.where(df[col_plus] > df[col_minus],
                                df[col_plus],
                                df[col_minus])

        # Step 2: if plus is NA → take minus
        df[col_global] = np.where(df[col_plus].isna(),
                                df[col_minus],
                                df[col_global])

        # Step 3: if minus is NA → take plus
        df[col_global] = np.where(df[col_minus].isna(),
                                df[col_plus],
                                df[col_global])
        

        df[col_global] = np.where(df[col_minus].isna() & df[col_plus].isna(),
                                np.nan,
                                df[col_global])

        # Step 4: replace -Inf with NaN
        df[col_global] = df[col_global].replace(-np.inf, np.nan)

        df[f"count{i}_global_dwXX"] = (
            (~df[f"count{i}_global_XX"].isna()) & (df[f"count{i}_global_XX"] > 0)
        ).astype(int)

        # Normalized delta
        if normalized:
            dict_glob[f"delta_D_{i}_global_XX"] = (
                (N1_new / (N1_new + N0_new)) * dict_glob.get(f"delta_D_{i}_in_XX", 0)
                + (N0_new / (N1_new + N0_new)) * dict_glob.get(f"delta_D_{i}_out_XX", 0)
            )

        # Number of switchers
        dict_glob[f"N_switchers_effect_{i}_XX"] = N1_new + N0_new
        dict_glob[f"N_switchers_effect_{i}_dwXX"] = (
            dict_glob.get(f"N1_dw_{i}_XX", 0) + dict_glob.get(f"N0_dw_{i}_XX", 0)
        )
        mat_res_XX[i - 1, 7] = dict_glob[f"N_switchers_effect_{i}_XX"]
        mat_res_XX[i - 1, 5] = dict_glob[f"N_switchers_effect_{i}_dwXX"]
        mat_res_XX[i - 1, 8] = i

        # Number of observations
        df[f"N_effect_{i}_XX"] = df[f"count{i}_global_XX"].sum(skipna=True)
        dict_glob[f"N_effect_{i}_XX"] = df[f"N_effect_{i}_XX"].mean(skipna=True)
        dict_glob[f"N_effect_{i}_dwXX"] = df[f"count{i}_global_dwXX"].sum(skipna=True)
        mat_res_XX[i - 1, 6] = dict_glob[f"N_effect_{i}_XX"]
        mat_res_XX[i - 1, 4] = int(dict_glob[f"N_effect_{i}_dwXX"])

        # Error check
        if dict_glob[f"N_switchers_effect_{i}_XX"] == 0 or dict_glob[f"N_effect_{i}_XX"] == 0:
            print(f"Effect {i} cannot be estimated. No switcher or control for this effect.")

        # DID computation
        df[f"DID_{i}_XX"] = df[f"U_Gg{i}_global_XX"].sum(skipna=True) / G_XX
        if normalized:
            df[f"DID_{i}_XX"] /= dict_glob[f"delta_D_{i}_global_XX"]
        dict_glob[f"DID_{i}_XX"] = df[f"DID_{i}_XX"].mean(skipna=True)

        # Missing check
        if (
            (switchers == "" and N1_new == 0 and N0_new == 0)
            or (switchers == "out" and N0_new == 0)
            or (switchers == "in" and N1_new == 0)
        ):
            dict_glob[f"DID_{i}_XX"] = np.nan

        mat_res_XX[i - 1, 0] = dict_glob[f"DID_{i}_XX"]


    # --------------------------
    # Average total effect
    # --------------------------
    U_Gg_den_plus_XX = np.nanmean(df["U_Gg_den_plus_XX"]) if "U_Gg_den_plus_XX" in df else 0
    U_Gg_den_minus_XX = np.nanmean(df["U_Gg_den_minus_XX"]) if "U_Gg_den_minus_XX" in df else 0
    U_Gg_den_plus_XX = 0 if np.isnan(U_Gg_den_plus_XX) else U_Gg_den_plus_XX
    U_Gg_den_minus_XX = 0 if np.isnan(U_Gg_den_minus_XX) else U_Gg_den_minus_XX

    if not trends_lin:
        if switchers == "":
            w_plus_XX = (
                            U_Gg_den_plus_XX * dict_glob['sum_N1_l_XX']
                            / (U_Gg_den_plus_XX * dict_glob['sum_N1_l_XX'] + 
                               U_Gg_den_minus_XX * dict_glob['sum_N0_l_XX'])
                        )
        elif switchers == "out":
            w_plus_XX = 0
        elif switchers == "in":
            w_plus_XX = 1

        df["U_Gg_global_XX"] = w_plus_XX * df["U_Gg_plus_XX"] + (1 - w_plus_XX) * df[
            "U_Gg_minus_XX"
        ]
        df.loc[df["first_obs_by_gp_XX"] == 0, "U_Gg_global_XX"] = np.nan

        df["delta_XX"] = df["U_Gg_global_XX"].sum(skipna=True) / G_XX
        delta_XX = df["delta_XX"].mean(skipna=True)
        dict_glob["Av_tot_effect"] = delta_XX

        mat_res_XX[l_XX, 0] = delta_XX
        N_switchers_effect_XX = sum(
            dict_glob.get(f"N_switchers_effect_{i}_XX", 0) for i in range(1, l_XX + 1)
        )
        N_switchers_effect_dwXX = sum(
            dict_glob.get(f"N_switchers_effect_{i}_dwXX", 0) for i in range(1, l_XX + 1)
        )
        mat_res_XX[l_XX, 7] = N_switchers_effect_XX
        mat_res_XX[l_XX, 5] = N_switchers_effect_dwXX
        mat_res_XX[l_XX, 8] = 0

        df["count_global_XX"] = 0
        for i in range(1, l_XX + 1):
            c = df[f"count{i}_global_XX"]
            df["count_global_XX"] = np.where(
                ~c.isna(), np.maximum(df["count_global_XX"], c), df["count_global_XX"]
            )
        df["count_global_dwXX"] = ((~df["count_global_XX"].isna()) & (df["count_global_XX"] > 0)).astype(int)
        N_effect_XX = df["count_global_XX"].sum(skipna=True)
        N_effect_dwXX = df["count_global_dwXX"].sum(skipna=True)
        mat_res_XX[l_XX, 6] = int(N_effect_XX)
        mat_res_XX[l_XX, 4] = int(N_effect_dwXX)


    # --------------------------
    # Placebos
    # --------------------------
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            df[f"U_Gg_pl_{i}_global_XX"] = (
                N1_pl_new / (N1_pl_new + N0_pl_new) * df[f"U_Gg_pl_{i}_plus_XX"]
                + N0_pl_new / (N1_pl_new + N0_pl_new) * df[f"U_Gg_pl_{i}_minus_XX"]
            )
            df.loc[df["first_obs_by_gp_XX"] == 0, f"U_Gg_pl_{i}_global_XX"] = np.nan

            # # Counts
            # df[f"count{i}_pl_global_XX"] = np.where(
            #     df[f"count{i}_pl_plus_XX"] > df[f"count{i}_pl_minus_XX"],
            #     df[f"count{i}_pl_plus_XX"],
            #     df[f"count{i}_pl_minus_XX"],
            # )
            # df[f"count{i}_pl_global_XX"] = np.where(
            #     df[f"count{i}_pl_plus_XX"].isna(),
            #     df[f"count{i}_pl_minus_XX"],
            #     df[f"count{i}_pl_global_XX"],
            # )
            # df[f"count{i}_pl_global_XX"] = np.where(
            #     df[f"count{i}_pl_minus_XX"].isna(),
            #     df[f"count{i}_pl_plus_XX"],
            #     df[f"count{i}_pl_global_XX"],
            # )
            # df.loc[df[f"count{i}_pl_global_XX"] == -np.inf, f"count{i}_pl_global_XX"] = np.nan

            col_plus   = f"count{i}_pl_plus_XX"
            col_minus  = f"count{i}_pl_minus_XX"
            col_global = f"count{i}_pl_global_XX"

            df[col_global] = df[[col_plus, col_minus]].max(axis=1)

            df[f"count{i}_pl_global_dwXX"] = (
                (~df[f"count{i}_pl_global_XX"].isna()) & (df[f"count{i}_pl_global_XX"] > 0)
            ).astype(int)

            # Normalized delta

            if normalized:
                N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
                N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)
                uax = dict_glob.get(f"delta_D_pl_{i}_out_XX", 0)
                print(f"N values {N1_pl_new} + { N0_pl_new}, {uax}")
                dict_glob[f"delta_D_pl_{i}_global_XX"] = (
                    (N1_pl_new / (N1_pl_new + N0_pl_new)) * dict_glob.get(f"delta_D_pl_{i}_in_XX", 0)
                    + (N0_pl_new / (N1_pl_new + N0_pl_new)) * dict_glob.get(f"delta_D_pl_{i}_out_XX", 0)
                )

            # DID placebo
            df[f"DID_placebo_{i}_XX"] = df[f"U_Gg_pl_{i}_global_XX"].sum(skipna=True) / G_XX
            if normalized:
                df[f"DID_placebo_{i}_XX"] /= dict_glob[f"delta_D_pl_{i}_global_XX"]
            dict_glob[f"DID_placebo_{i}_XX"] = df[f"DID_placebo_{i}_XX"].mean(skipna=True)

            # Missing check
            if (
                (switchers == "" and N1_pl_new == 0 and N0_pl_new == 0)
                or (switchers == "out" and N0_pl_new == 0)
                or (switchers == "in" and N1_pl_new == 0)
            ):
                dict_glob[f"DID_placebo_{i}_XX"] = np.nan

            mat_res_XX[l_XX + i, 0] = dict_glob[f"DID_placebo_{i}_XX"]

            # Number of switchers
            dict_glob[f"N_switchers_placebo_{i}_XX"] = N1_pl_new + N0_pl_new
            dict_glob[f"N_switchers_placebo_{i}_dwXX"] = dict_glob.get(f"N1_dw_placebo_{i}_XX", 0) + dict_glob.get(f"N0_dw_placebo_{i}_XX", 0)
            mat_res_XX[l_XX + i, 7] = dict_glob[f"N_switchers_placebo_{i}_XX"]
            mat_res_XX[l_XX + i, 5] = dict_glob[f"N_switchers_placebo_{i}_dwXX"]
            mat_res_XX[l_XX + i, 8] = -i

            df[f"N_placebo_{i}_XX"] = df[f"count{i}_pl_global_XX"].sum(skipna=True)
            dict_glob[f"N_placebo_{i}_XX"] = df[f"N_placebo_{i}_XX"].mean(skipna=True)
            mat_res_XX[l_XX + i, 6] = dict_glob[f"N_placebo_{i}_XX"]
            mat_res_XX[l_XX + i, 4] = int(df[f"count{i}_pl_global_dwXX"].sum(skipna=True))

            if dict_glob[f"N_switchers_placebo_{i}_XX"] == 0 or dict_glob[f"N_placebo_{i}_XX"] == 0:
                print(f"Placebo {i} cannot be estimated. No switcher or control for this placebo.")



    

    # Patch significance level
    ci_level = ci_level / 100
    z_level = norm.ppf(ci_level + (1 - ci_level) / 2)

    # Ensure df is a pandas DataFrame
    df = pd.DataFrame(df)

    # Loop over effects
    for i in range(1, l_XX + 1):
        N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
        N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)

        valid_case = (
            (switchers == "" and (N1_new != 0 or N0_new != 0))
            or (switchers == "out" and N0_new != 0)
            or (switchers == "in" and N1_new != 0)
        )

        if valid_case:
            # Aggregate U_Gg_var for switchers in and out
            df[f"U_Gg_var_glob_{i}_XX"] = (
                df[f"U_Gg_var_{i}_in_XX"] * (N1_new / (N1_new + N0_new))
                + df[f"U_Gg_var_{i}_out_XX"] * (N0_new / (N1_new + N0_new))
            )

            if cluster is None:
                # Without clustering
                df[f"U_Gg_var_glob_eff{i}_sqrd_XX"] = (
                    df[f"U_Gg_var_glob_{i}_XX"] ** 2 * df["first_obs_by_gp_XX"]
                )
                dict_glob[f"sum_for_var_{i}_XX"] = (
                    df[f"U_Gg_var_glob_eff{i}_sqrd_XX"].sum(skipna=True) / (G_XX**2)
                )
            else:

                # Base column names
                col_base = f"U_Gg_var_glob_{i}_XX"
                col_clust = f"clust_U_Gg_var_glob_{i}_XX"
                col_clust_sq = f"clust_U_Gg_var_glob_{i}_2_XX"

                # 1. Multiply U_Gg_var_glob by first_obs_by_gp_XX
                df[col_base] = df[col_base] * df["first_obs_by_gp_XX"]

                # 2. Sum within cluster → total(U_Gg_var_glob_i_XX)
                df[col_clust] = df.groupby('cluster_XX')[col_base].transform("sum")

                # 3. Compute average of square (only first obs per cluster)
                df[col_clust_sq] = (df[col_clust] ** 2) * df["first_obs_by_clust_XX"]

                # 4. Scalar sum_for_var = sum / G_XX^2
                if G_XX is not None:
                    dict_glob[f"sum_for_var_{i}_XX"] = df[col_clust_sq].sum(skipna=True) / (G_XX ** 2)
                else:
                    dict_glob[f"sum_for_var_{i}_XX"] = np.nan  # keep undefined if G_XX not provided

                # 5. Replace U_Gg_var_glob with cluster total
                df[col_base] = df[col_clust]

            # Compute SE
            dict_glob[f"se_{i}_XX"] = np.sqrt(dict_glob[f"sum_for_var_{i}_XX"])

            # Normalize SE if requested
            if normalized:
                dict_glob[f"se_{i}_XX"] /= dict_glob.get(f"delta_D_{i}_global_XX", 1)

            # Store results
            mat_res_XX[i - 1, 1] = dict_glob[f"se_{i}_XX"]
            dict_glob[f"se_effect_{i}"] = dict_glob[f"se_{i}_XX"]

            dict_glob[f"LB_CI_{i}_XX"] = dict_glob[f"DID_{i}_XX"] - z_level * dict_glob[f"se_{i}_XX"]
            mat_res_XX[i - 1, 2] = dict_glob[f"LB_CI_{i}_XX"]

            dict_glob[f"UB_CI_{i}_XX"] = dict_glob[f"DID_{i}_XX"] + z_level * dict_glob[f"se_{i}_XX"]
            mat_res_XX[i - 1, 3] = dict_glob[f"UB_CI_{i}_XX"]



    # ----------------------------------------
    # Variances of placebo estimators
    # ----------------------------------------
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            valid_case = (
                (switchers == "" and (N1_pl_new != 0 or N0_pl_new != 0))
                or (switchers == "out" and N0_pl_new != 0)
                or (switchers == "in" and N1_pl_new != 0)
            )

            if valid_case:
                # Aggregate U_Gg_var for placebo
                df[f"U_Gg_var_glob_pl_{i}_XX"] = (
                    df[f"U_Gg_var_pl_{i}_in_XX"] * (N1_pl_new / (N1_pl_new + N0_pl_new))
                    + df[f"U_Gg_var_pl_{i}_out_XX"] * (N0_pl_new / (N1_pl_new + N0_pl_new))
                )

                if cluster is None:
                    df[f"U_Gg_var_glob_pl_{i}_2_XX"] = (
                        df[f"U_Gg_var_glob_pl_{i}_XX"] ** 2 * df["first_obs_by_gp_XX"]
                    )
                    dict_glob[f"sum_for_var_placebo_{i}_XX"] = (
                        df[f"U_Gg_var_glob_pl_{i}_2_XX"].sum(skipna=True) / (G_XX**2)
                    )
                else:

                    # Compute average of square
                    # Assume df is a pandas DataFrame and i, G_XX are defined
                    col_clust = f"clust_U_Gg_var_glob_pl_{i}_XX"
                    col_clust_sq = f"clust_U_Gg_var_glob_pl_{i}_2_XX"
                    col_U = f"U_Gg_var_glob_pl_{i}_XX"

                    df[col_U] = (
                        df[col_U] * df["first_obs_by_gp_XX"]
                    )

                    # Sum within cluster
                    df[col_clust] = (
                            df.groupby('cluster_XX')[col_U]
                            .transform(lambda x: x.sum(skipna=True))
                        )

                    # 1. Compute square * first_obs_by_clust_XX
                    df[col_clust_sq] = df[col_clust]**2 * df["first_obs_by_clust_XX"]

                    # 2. Compute average (like assign in R)
                    dict_glob[f"sum_for_var_placebo_{i}_XX"] = df[col_clust_sq].sum(skipna=True) / (G_XX**2)

                    # 3. Copy clust_ column into U_ column
                    df[col_U] = df[col_clust]

                # SE
                dict_glob[f"se_placebo_{i}_XX"] = np.sqrt(
                    dict_glob[f"sum_for_var_placebo_{i}_XX"]
                )
                if normalized:
                    dict_glob[f"se_placebo_{i}_XX"] /= dict_glob.get(
                        f"delta_D_pl_{i}_global_XX", 1
                    )

                mat_res_XX[l_XX + i, 1] = dict_glob[f"se_placebo_{i}_XX"]
                dict_glob[f"se_placebo_{i}"] = dict_glob[f"se_placebo_{i}_XX"]

                dict_glob[f"LB_CI_placebo_{i}_XX"] = dict_glob[f"DID_placebo_{i}_XX"] - z_level * dict_glob[f"se_placebo_{i}_XX"]
                mat_res_XX[l_XX + i, 2] = dict_glob[f"LB_CI_placebo_{i}_XX"]

                dict_glob[f"UB_CI_placebo_{i}_XX"] = dict_glob[f"DID_placebo_{i}_XX"] + z_level * dict_glob[f"se_placebo_{i}_XX"]
                mat_res_XX[l_XX + i, 3] = dict_glob[f"UB_CI_placebo_{i}_XX"]


    # ----------------------------------------
    # Variance of average total effect
    # ----------------------------------------
    if not trends_lin:
        valid_case = (
            (switchers == "" and (dict_glob['sum_N1_l_XX'] != 0 or dict_glob['sum_N0_l_XX'] != 0))
            or (switchers == "out" and dict_glob['sum_N0_l_XX'] != 0)
            or (switchers == "in" and dict_glob['sum_N1_l_XX'] != 0)
        )

        if valid_case:
            df["U_Gg_var_global_XX"] = (
                w_plus_XX * df["U_Gg_var_plus_XX"]
                + (1 - w_plus_XX) * df["U_Gg_var_minus_XX"]
            )

            if cluster is None:
                df["U_Gg_var_global_2_XX"] = df["U_Gg_var_global_XX"] ** 2 * df["first_obs_by_gp_XX"]
                sum_for_var_XX = df["U_Gg_var_global_2_XX"].sum(skipna=True) / (G_XX**2)
            else:
                df["U_Gg_var_global_XX"] = df["U_Gg_var_global_XX"] * df["first_obs_by_gp_XX"]
                df["clust_U_Gg_var_global_XX"] = df.groupby('cluster_XX')["U_Gg_var_global_XX"].transform(lambda x: x.sum(skipna=True))
                df["clust_U_Gg_var_global_XX"] = df["clust_U_Gg_var_global_XX"] ** 2 * df["first_obs_by_clust_XX"]
                sum_for_var_XX = df["clust_U_Gg_var_global_XX"].sum(skipna=True) / (G_XX**2)

            se_XX = np.sqrt(sum_for_var_XX)
            mat_res_XX[l_XX, 1] = se_XX
            dict_glob["se_avg_total_effect"] = se_XX

            LB_CI_XX = delta_XX - z_level * se_XX
            mat_res_XX[l_XX, 2] = LB_CI_XX
            UB_CI_XX = delta_XX + z_level * se_XX
            mat_res_XX[l_XX, 3] = UB_CI_XX



    # ----------------------------------------
    # Average number of cumulated effects
    # ----------------------------------------
    for i in range(1, l_XX + 1):
        col = f"delta_D_g_{i}_XX"
        if col in df:
            df.drop(columns=[col], inplace=True)

    df["M_g_XX"] = np.where(
        l_XX <= df["T_g_XX"] - df["F_g_XX"] + 1,
        l_XX,
        df["T_g_XX"] - df["F_g_XX"] + 1,
    )



    # Build delta_D_g_XX
    df["delta_D_g_XX"] = 0
    for j in range(1, l_XX + 1):
        df["delta_D_g_XX_temp"] = np.where(
            df[f"delta_D_g_{j}_plus_XX"] != 0,
            df[f"delta_D_g_{j}_plus_XX"],
            df[f"delta_D_g_{j}_minus_XX"],
        )
        df["delta_D_g_XX_temp"] = df["delta_D_g_XX_temp"].replace(0, np.nan)
        df["delta_D_g_XX"] = np.where(
            df["switchers_tag_XX"] == j,
            df["delta_D_g_XX"] + df["delta_D_g_XX_temp"],
            df["delta_D_g_XX"],
        )

    df["delta_D_g_num_XX"] = df["delta_D_g_XX"] * (df["M_g_XX"] - (df["switchers_tag_XX"] - 1))
    delta_D_num_total = df["delta_D_g_num_XX"].sum(skipna=True)
    delta_D_denom_total = df["delta_D_g_XX"].sum(skipna=True)
    delta_D_avg_total = delta_D_num_total / delta_D_denom_total


    # ----------------------------------------
    # Cluster adjustment
    # ----------------------------------------
    if cluster is not None:
        df["first_obs_by_gp_XX"] = df["first_obs_by_clust_XX"]

    # --------------------------------------------------------------------------------
    ###### Performing a test to see whether all effects are jointly equal to 0
    # --------------------------------------------------------------------------------
    

    all_Ns_not_zero = np.nan
    all_delta_not_zero = np.nan
    p_jointeffects = None

    # Test can only be run when at least two effects requested
    if l_XX != 0 and l_XX > 1:
        all_Ns_not_zero = 0
        all_delta_not_zero = 0

        # Count number of estimable effects
        for i in range(1, l_XX + 1):
            N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
            N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)

            if (
                (switchers == "" and (N1_new != 0 or N0_new != 0))
                or (switchers == "out" and N0_new != 0)
                or (switchers == "in" and N1_new != 0)
            ):
                all_Ns_not_zero += 1

            if normalized:
                delta_val = dict_glob.get(f"delta_D_{i}_global_XX", np.nan)
                if delta_val != 0 and not np.isnan(delta_val):
                    all_delta_not_zero += 1

        # Test feasible only if all requested effects were computed
        feasible = (
            (all_Ns_not_zero == l_XX and not normalized)
            or (normalized and all_Ns_not_zero == l_XX and all_delta_not_zero == l_XX)
        )

        if feasible:
            # Collect DID estimates and variances
            didmgt_Effects = np.zeros((l_XX, 1))
            didmgt_Var_Effects = np.zeros((l_XX, l_XX))

            for i in range(1, l_XX + 1):
                didmgt_Effects[i - 1, 0] = dict_glob.get(f"DID_{i}_XX", 0)
                didmgt_Var_Effects[i - 1, i - 1] = dict_glob.get(f"se_{i}_XX", 0) ** 2

                if i < l_XX:
                    for j in range(i + 1, l_XX + 1):
                        if not normalized:
                            df[f"U_Gg_var_{i}_{j}_XX"] = (
                                df[f"U_Gg_var_glob_{i}_XX"] + df[f"U_Gg_var_glob_{j}_XX"]
                            )
                        else:
                            df[f"U_Gg_var_{i}_{j}_XX"] = (
                                df[f"U_Gg_var_glob_{i}_XX"] / dict_glob.get(f"delta_D_{i}_global_XX", 1)
                                + df[f"U_Gg_var_glob_{j}_XX"] / dict_glob.get(f"delta_D_{j}_global_XX", 1)
                            )

                        df[f"U_Gg_var_{i}_{j}_2_XX"] = (
                            df[f"U_Gg_var_{i}_{j}_XX"] ** 2 * df["first_obs_by_gp_XX"]
                        )

                        var_sum = df[f"U_Gg_var_{i}_{j}_2_XX"].sum(skipna=True) / (G_XX**2)
                        se_i = dict_glob.get(f"se_{i}_XX", 0)
                        se_j = dict_glob.get(f"se_{j}_XX", 0)

                        cov_ij = (var_sum - se_i**2 - se_j**2) / 2
                        dict_glob[f"cov_{i}_{j}_XX"] = cov_ij

                        didmgt_Var_Effects[i - 1, j - 1] = cov_ij
                        didmgt_Var_Effects[j - 1, i - 1] = cov_ij

            # Inverse covariance matrix (pseudo-inverse if singular)
            didmgt_Var_Effects_inv = np.linalg.pinv(didmgt_Var_Effects)

            # Wald χ² statistic
            didmgt_chi2effects = (
                didmgt_Effects.T @ didmgt_Var_Effects_inv @ didmgt_Effects
            )

            # p-value
            p_jointeffects = 1 - chi2.cdf(didmgt_chi2effects[0, 0], df=l_XX)

        else:
            p_jointeffects = np.nan
            print(
                "Some effects could not be estimated. Therefore, the test of joint Noneity of the effects could not be computed."
            )


    # --------------------------------------------------------------------------------
    ###### Performing a test to see whether all placebos are jointly equal to 0
    # --------------------------------------------------------------------------------

    

    all_Ns_pl_not_zero = np.nan
    all_delta_pl_not_zero = np.nan
    p_jointplacebo = None

    # Test can only be run when at least two placebos requested
    if l_placebo_XX != 0 and l_placebo_XX > 1:
        all_Ns_pl_not_zero = 0
        all_delta_pl_not_zero = 0

        # Count number of estimable placebos
        for i in range(1, int(l_placebo_XX) + 1):
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            if (
                (switchers == "" and (N1_pl_new != 0 or N0_pl_new != 0))
                or (switchers == "out" and N0_pl_new != 0)
                or (switchers == "in" and N1_pl_new != 0)
            ):
                all_Ns_pl_not_zero += 1

            if normalized:
                delta_val = dict_glob.get(f"delta_D_pl_{i}_global_XX", np.nan)
                if delta_val != 0 and not np.isnan(delta_val):
                    all_delta_pl_not_zero += 1

        # Test feasible only if all requested placebos were computed
        feasible = (
            (all_Ns_pl_not_zero == l_placebo_XX and not normalized)
            or (normalized and all_Ns_pl_not_zero == l_placebo_XX and all_delta_pl_not_zero == l_placebo_XX)
        )

        if feasible:
            # Collect DID placebo estimates and variances
            didmgt_Placebo = np.zeros((int(l_placebo_XX), 1))
            didmgt_Var_Placebo = np.zeros((int(l_placebo_XX), int(l_placebo_XX)))

            for i in range(1, int(l_placebo_XX) + 1):
                didmgt_Placebo[i - 1, 0] = dict_glob.get(f"DID_placebo_{i}_XX", 0)
                didmgt_Var_Placebo[i - 1, i - 1] = dict_glob.get(f"se_placebo_{i}_XX", 0) ** 2

                if i < int(l_placebo_XX):
                    for j in range(i + 1, int(l_placebo_XX) + 1):
                        if not normalized:
                            df[f"U_Gg_var_pl_{i}_{j}_XX"] = (
                                df[f"U_Gg_var_glob_pl_{i}_XX"] + df[f"U_Gg_var_glob_pl_{j}_XX"]
                            )
                        else:
                            df[f"U_Gg_var_pl_{i}_{j}_XX"] = (
                                df[f"U_Gg_var_glob_pl_{i}_XX"] / dict_glob.get(f"delta_D_pl_{i}_global_XX", 1)
                                + df[f"U_Gg_var_glob_pl_{j}_XX"] / dict_glob.get(f"delta_D_pl_{j}_global_XX", 1)
                            )

                        df[f"U_Gg_var_pl_{i}_{j}_2_XX"] = (
                            df[f"U_Gg_var_pl_{i}_{j}_XX"] ** 2 * df["first_obs_by_gp_XX"]
                        )

                        var_sum = df[f"U_Gg_var_pl_{i}_{j}_2_XX"].sum(skipna=True) / (G_XX**2)
                        se_i = dict_glob.get(f"se_placebo_{i}_XX", 0)
                        se_j = dict_glob.get(f"se_placebo_{j}_XX", 0)

                        cov_ij = (var_sum - se_i**2 - se_j**2) / 2
                        dict_glob[f"cov_pl_{i}_{j}_XX"] = cov_ij

                        didmgt_Var_Placebo[i - 1, j - 1] = cov_ij
                        didmgt_Var_Placebo[j - 1, i - 1] = cov_ij

            # Inverse covariance matrix (pseudo-inverse if singular)
            didmgt_Var_Placebo_inv = Ginv(didmgt_Var_Placebo)

            # Wald χ² statistic
            didmgt_chi2placebo = didmgt_Placebo.T @ didmgt_Var_Placebo_inv @ didmgt_Placebo

            # p-value
            p_jointplacebo = 1 - chi2.cdf(didmgt_chi2placebo[0, 0], df=l_placebo_XX)

        else:
            p_jointplacebo = np.nan
            print(
                "Some placebos could not be estimated. Therefore, the test of joint Noneity of the placebos could not be computed."
            )

    het_res = pd.DataFrame()

    if predict_het is not None and len(predict_het_good) > 0:
        # Define which effects to calculate
        if -1 in het_effects:
            het_effects = list(range(1, l_XX + 1))
        all_effects_XX = [i for i in range(1, l_XX + 1) if i in het_effects]

        if any(np.isnan(all_effects_XX)):
            raise ValueError(
                "Error in predict_het second argument: please specify only numbers ≤ number of effects requested"
            )

        # Preliminaries: Yg, Fg-1
        df["Yg_Fg_min1_XX"] = np.where(
            df["time_XX"] == df["F_g_XX"] - 1, df["outcome_non_diff_XX"], np.nan
        )
        df["Yg_Fg_min1_XX"] = df.groupby("group_XX")["Yg_Fg_min1_XX"].transform("mean")
        df["feasible_het_XX"] = ~df["Yg_Fg_min1_XX"].isna()

        if trends_lin is not None:
            df["Yg_Fg_min2_XX"] = np.where(
                df["time_XX"] == df["F_g_XX"] - 2, df["outcome_non_diff_XX"], np.nan
            )
            df["Yg_Fg_min2_XX"] = df.groupby("group_XX")["Yg_Fg_min2_XX"].transform("mean")
            df["Yg_Fg_min2_XX"] = df["Yg_Fg_min2_XX"].replace({np.nan: None})
            df["feasible_het_XX"] &= ~df["Yg_Fg_min2_XX"].isna()

        # Order and group index
        df = df.sort_values(["group_XX", "time_XX"])
        df["gr_id"] = df.groupby("group_XX").cumcount() + 1

        lhyp = [f"{v}=0" for v in predict_het_good]

        # Loop over requested effects
        for i in all_effects_XX:
            # Sample restriction
            het_sample = df.loc[
                (df["F_g_XX"] - 1 + i <= df["T_g_XX"]) & (df["feasible_het_XX"])
            ].copy()

            # Yg, Fg-1 + i
            df[f"Yg_Fg_{i}_XX"] = np.where(
                df["time_XX"] == df["F_g_XX"] - 1 + i, df["outcome_non_diff_XX"], np.nan
            )
            df[f"Yg_Fg_{i}_XX"] = df.groupby("group_XX")[f"Yg_Fg_{i}_XX"].transform("mean")

            df["diff_het_XX"] = df[f"Yg_Fg_{i}_XX"] - df["Yg_Fg_min1_XX"]
            if trends_lin:
                df["diff_het_XX"] -= i * (df["Yg_Fg_min1_XX"] - df["Yg_Fg_min2_XX"])

            # Interaction term
            df[f"prod_het_{i}_XX"] = df["S_g_het_XX"] * df["diff_het_XX"]
            df.loc[df["gr_id"] != 1, f"prod_het_{i}_XX"] = np.nan

            # Regression formula
            het_reg = f"prod_het_{i}_XX ~ {' + '.join(predict_het_good)}"

            # Add categorical dummies
            for v in ["F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam]:
                if het_sample[v].nunique() > 1:
                    het_reg += f" + C({v})"

            # Run regression with robust SE (HC1)
            model = smf.wls(het_reg, data=het_sample, weights=het_sample["weight_XX"]).fit(
                cov_type="HC1"
            )

            # Extract results
            coefs = model.params[predict_het_good]
            ses = model.bse[predict_het_good]
            ts = model.tvalues[predict_het_good]

            t_stat = student_t.ppf(0.975, model.df_resid)
            lb = coefs - t_stat * ses
            ub = coefs + t_stat * ses

            f_test = model.f_test(lhyp)
            f_stat = f_test.pvalue

            # Append to het_res
            het_res = pd.concat(
                [
                    het_res,
                    pd.DataFrame(
                        {
                            "effect": i,
                            "covariate": predict_het_good,
                            "Estimate": coefs.values,
                            "SE": ses.values,
                            "t": ts.values,
                            "LB": lb.values,
                            "UB": ub.values,
                            "N": [int(model.nobs)] * len(predict_het_good),
                            "pF": [f_stat] * len(predict_het_good),
                        }
                    ),
                ],
                ignore_index=True,
            )

        het_res = het_res.sort_values(["covariate", "effect"])

    # ----------------------------
    # Test that all DID_l effects are equal
    # ----------------------------
    if effects_equal and l_XX > 1:
        all_Ns_not_zero = 0
        for i in range(1, l_XX + 1):
            N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
            N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
            if (
                (switchers == "" and (N1_new != 0 or N0_new != 0))
                or (switchers == "out" and N0_new != 0)
                or (switchers == "in" and N1_new != 0)
            ):
                all_Ns_not_zero += 1

        if all_Ns_not_zero == l_XX:
            didmgt_Effects = mat_res_XX[:l_XX, 0]
            didmgt_Var_Effects = np.zeros((l_XX, l_XX))
            didmgt_identity = np.zeros((l_XX - 1, l_XX))

            for i in range(1, l_XX + 1):
                N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
                N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
                if (
                    (switchers == "" and (N1_new != 0 or N0_new != 0))
                    or (switchers == "out" and N0_new != 0)
                    or (switchers == "in" and N1_new != 0)
                ):
                    didmgt_Var_Effects[i - 1, i - 1] = dict_glob.get(f"se_{i}_XX", 0) ** 2
                    if i < l_XX:
                        didmgt_identity[i - 1, i - 1] = 1

                    if i < l_XX:
                        for j in range(i + 1, l_XX + 1):
                            if not normalized:
                                df[f"U_Gg_var_{i}_{j}_XX"] = (
                                    df[f"U_Gg_var_glob_{i}_XX"] + df[f"U_Gg_var_glob_{j}_XX"]
                                )
                            else:
                                df[f"U_Gg_var_{i}_{j}_XX"] = (
                                    df[f"U_Gg_var_glob_{i}_XX"] / dict_glob.get(f"delta_D_{i}_global_XX", 1)
                                    + df[f"U_Gg_var_glob_{j}_XX"] / dict_glob.get(f"delta_D_{j}_global_XX", 1)
                                )

                            df[f"U_Gg_var_{i}_{j}_2_XX"] = (
                                df[f"U_Gg_var_{i}_{j}_XX"] ** 2 * df["first_obs_by_gp_XX"]
                            )
                            var_sum = df[f"U_Gg_var_{i}_{j}_2_XX"].sum(skipna=True) / (G_XX**2)
                            cov_ij = (var_sum - dict_glob.get(f"se_{i}_XX", 0) ** 2 - dict_glob.get(f"se_{j}_XX", 0) ** 2) / 2
                            dict_glob[f"cov_{i}_{j}_XX"] = cov_ij

                            didmgt_Var_Effects[i - 1, j - 1] = cov_ij
                            didmgt_Var_Effects[j - 1, i - 1] = cov_ij

            # Demeaned effects: test equality
            didmgt_D = didmgt_identity - np.full((l_XX - 1, l_XX), 1 / l_XX)
            didmgt_test_effects = didmgt_D @ didmgt_Effects
            didmgt_test_var = didmgt_D @ didmgt_Var_Effects @ didmgt_D.T
            # enforce symmetry
            didmgt_test_var = (didmgt_test_var + didmgt_test_var.T) / 2

            # Wald χ² statistic
            didmgt_chi2_equal_ef = (
                didmgt_test_effects.T @ np.linalg.pinv(didmgt_test_var) @ didmgt_test_effects
            )
            p_equality_effects = 1 - chi2.cdf(didmgt_chi2_equal_ef[0], df=l_XX - 1)
            dict_glob["p_equality_effects"] = p_equality_effects
        else:
            print(
                "Some effects could not be estimated. Therefore, the test of equality of effects could not be computed."
            )

    # assume df is a pandas DataFrame
    # assume l_XX, l_placebo_XX, normalized, G_XX, mat_res_XX are already defined

    # 1. Total length
    l_tot_XX = l_XX + l_placebo_XX

    # 2. Initialize covariance matrix with NaNs
    didmgt_vcov = np.full((int(l_tot_XX), int(l_tot_XX)), np.nan)

    # 3. Build row/col names
    mat_names = [
        f"Effect_{i}" if i <= l_XX else f"Placebo_{i - l_XX}"
        for i in range(1, int(l_tot_XX) + 1)
    ]

    # Optionally wrap in DataFrame for labeled covariance matrix
    didmgt_vcov = pd.DataFrame(didmgt_vcov, index=mat_names, columns=mat_names)

    # 4. Loop for effects
    for i in range(1, l_XX + 1):
        col_glob = f"U_Gg_var_glob_{i}_XX"
        col_comb = f"U_Gg_var_comb_{i}_XX"

        if not normalized:
            df[col_comb] = df[col_glob] if col_glob in df else np.nan
        else:
            delta_name = f"delta_D_{i}_global_XX"
            df[col_comb] = (
                df[col_glob] / dict_glob[delta_name] if col_glob in df else np.nan
            )

    # 5. Loop for placebos
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            col_glob_pl = f"U_Gg_var_glob_pl_{i}_XX"
            col_comb = f"U_Gg_var_comb_{l_XX + i}_XX"

            if not normalized:
                df[col_comb] = df[col_glob_pl] if col_glob_pl in df else np.nan
            else:
                delta_name = f"delta_D_pl_{i}_global_XX"
                df[col_comb] = (
                    df[col_glob_pl] / dict_glob[delta_name]
                    if col_glob_pl in df
                    else np.nan
                )

    # 6. Fill the covariance matrix
    for i in range(1, int(l_tot_XX) + 1):
        didmgt_vcov.iloc[i - 1, i - 1] = mat_res_XX[i + (i > l_XX) - 1, 1] ** 2

        j = 1
        while j < i:
            col_i = f"U_Gg_var_comb_{i}_XX"
            col_j = f"U_Gg_var_comb_{j}_XX"
            col_temp = f"U_Gg_var_comb_{i}_{j}_2_XX"

            df[col_temp] = (df[col_i] + df[col_j]) ** 2 * df["first_obs_by_gp_XX"]

            var_temp = df[col_temp].sum(skipna=True) / (G_XX ** 2)

            didmgt_vcov.iloc[i - 1, j - 1] = didmgt_vcov.iloc[j - 1, i - 1] = (
                var_temp
                - mat_res_XX[i + (i > l_XX) - 1, 1] ** 2
                - mat_res_XX[j + (j > l_XX) - 1, 1] ** 2
            ) / 2

            # cleanup temporary column
            df.drop(columns=[col_temp], inplace=True)

            j += 1

    # -------------------------------------
    # Format results matrix
    # -------------------------------------

    rownames_arr = np.array(rownames)
    colnames_arr = [
        "Estimate", "SE", "LB CI", "UB CI",
        "N", "Switchers", "N.w", "Switchers.w", "Time"
    ]

    mat_res_df = pd.DataFrame(mat_res_XX)
    mat_res_df.columns = colnames_arr 

    # Save results if requested
    if save_results is not None:
        mat_res_df.to_csv(save_results, index=True)

    # -------------------------------------
    # Separate Effect matrix and ATE matrix
    # -------------------------------------
    Effect_mat = mat_res_df.iloc[:l_XX, :-1].copy()
    ATE_mat = mat_res_df.iloc[[l_XX], :-1].copy()
    ATE_mat.index = ["Average_Total_Effect"]
    Effect_mat.index = mat_names[:l_XX]


    # -------------------------------------
    # Assemble did_multiplegt_dyn
    # -------------------------------------
    out_names = [
        "N_Effects", "N_Placebos", "Effects", "ATE",
        "delta_D_avg_total", "max_pl", "max_pl_gap"
    ]

    did_multiplegt_dyn = [
        l_XX,
        int(l_placebo_XX),
        Effect_mat,
        ATE_mat,
        delta_D_avg_total,
        max_pl_XX,
        max_pl_gap_XX,
    ]

    if p_jointeffects is not None:
        did_multiplegt_dyn.append(p_jointeffects)
        out_names.append("p_jointeffects")

    if effects_equal:
        did_multiplegt_dyn.append(p_equality_effects)
        out_names.append("p_equality_effects")

    if placebo != 0:
        Placebo_mat = mat_res_df.iloc[(l_XX + 1):, :-1].copy()
        did_multiplegt_dyn.append(Placebo_mat)
        out_names.append("Placebos")

        if placebo > 1 and l_placebo_XX > 1:
            did_multiplegt_dyn.append(p_jointplacebo)
            out_names.append("p_jointplacebo")

    if predict_het is not None and len(predict_het_good) > 0:
        did_multiplegt_dyn.append(het_res)
        out_names.append("predict_het")

    # optional debugging
    # did_multiplegt_dyn.append(df)
    # out_names.append("debug")

    # Name results
    did_multiplegt_dyn = dict(zip(out_names, did_multiplegt_dyn))

    # -------------------------------------
    # Collect delta if normalized
    # -------------------------------------
    delta = {}
    if normalized:
        for i in range(1, l_XX + 1):
            delta[f"delta_D_{i}_global_XX"] = dict_glob.get(f"delta_D_{i}_global_XX")

    # -------------------------------------
    # Collect coefficients and vcov
    # -------------------------------------
    coef = {
        "b": mat_res_df.iloc[np.r_[0:l_XX, (l_XX + 1):], 0].values,
        "vcov": didmgt_vcov,
    }

    # -------------------------------------
    # Assemble return object
    # -------------------------------------
    ret = {
        "df": df,
        "did_multiplegt_dyn": did_multiplegt_dyn,
        "delta": delta,
        "l_XX": l_XX,
        "T_max_XX": T_max_XX,
        "mat_res_XX": mat_res_df,
        'dict_glob' : dict_glob
    }

    if placebo != 0:
        ret["l_placebo_XX"] = l_placebo_XX

    ret["coef"] = coef

    return( ret )