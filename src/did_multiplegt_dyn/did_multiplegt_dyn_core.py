import numpy as np
import pandas as pd

def compute_E_hat_gt_with_nans(df, i, type_sect = "effect"):
    
    if type_sect == "effect":
            
        E_hat   = f"E_hat_gt_{i}_XX"
        mean_ns = f"mean_cohort_{i}_ns_t_XX"
        mean_s  = f"mean_cohort_{i}_s_t_XX"
        mean_nss = f"mean_cohort_{i}_ns_s_t_XX"

        dof_ns  = f"dof_cohort_{i}_ns_t_XX"
        dof_s   = f"dof_cohort_{i}_s_t_XX"
        dof_nss = f"dof_cohort_{i}_ns_s_t_XX"
    
    else:

        E_hat   = f"E_hat_gt_pl_{i}_XX"
        mean_ns = f"mean_cohort_pl_{i}_ns_t_XX"
        mean_s  = f"mean_cohort_pl_{i}_s_t_XX"
        mean_nss = f"mean_cohort_pl_{i}_ns_s_t_XX"

        dof_ns  = f"dof_cohort_pl_{i}_ns_t_XX"
        dof_s   = f"dof_cohort_pl_{i}_s_t_XX"
        dof_nss = f"dof_cohort_pl_{i}_ns_s_t_XX"

    # nanmask = df[dof_s].notnull() | df[dof_ns].notnull() | df[dof_nss].notnull()
    # Start missing
    df[E_hat] = np.nan

    # Conditions (same as your Stata code)
    cond_A = (df["time_XX"] < df["F_g_XX"]) | (df["F_g_XX"] - 1 + i == df["time_XX"])
    cond_B = (df["time_XX"] < df["F_g_XX"]) & (df[dof_ns].replace(np.nan, 9999) >= 2)
    cond_C = (df["F_g_XX"] - 1 + i == df["time_XX"]) & (df[dof_s].replace(np.nan, 9999) >= 2)
    cond_D = (
        (df[dof_nss] >= 2)
        & (
            ((df["F_g_XX"] - 1 + i == df["time_XX"]) & (df[dof_s].replace(np.nan, 9999) == 1))
            | ((df["time_XX"] < df["F_g_XX"]) & (df[dof_ns].replace(np.nan, 9999) == 1))
        )
    )

    # gen E_hat = 0 if A  (but only where constructing vars are fully observed)
    df.loc[cond_A, E_hat] = 0.0

    # replace with mean_ns if B
    df.loc[cond_B, E_hat] = df.loc[cond_B, mean_ns]
    df.loc[df[mean_ns].isna(), E_hat] = np.nan

    # replace with mean_s if C
    df.loc[cond_C, E_hat] = df.loc[cond_C, mean_s]

    # replace with mean_nss if D
    df.loc[cond_D, E_hat] = df.loc[cond_D, mean_nss]
    df.loc[df[mean_nss].isna(), E_hat] = np.nan
    
    return df

def compute_DOF_gt_with_nans(df, i, type_sect = "effect"):

    if type_sect == "effect":
        
        DOF     = f"DOF_gt_{i}_XX"
        dof_s   = f"dof_cohort_{i}_s_t_XX"
        dof_ns  = f"dof_cohort_{i}_ns_t_XX"
        dof_nss = f"dof_cohort_{i}_ns_s_t_XX"
    else:
        DOF     = f"DOF_gt_pl_{i}_XX"
        dof_s   = f"dof_cohort_pl_{i}_s_t_XX"
        dof_ns  = f"dof_cohort_pl_{i}_ns_t_XX"
        dof_nss = f"dof_cohort_pl_{i}_ns_s_t_XX"
        
    # Start missing, like Stata
    df = df.reset_index(drop = True)
    df[DOF] = np.nan

    # Conditions
    cond_A = (df["time_XX"] < df["F_g_XX"]) | ((df["F_g_XX"] - 1 + i) == df["time_XX"])
    cond_B = ((df["F_g_XX"] - 1 + i) == df["time_XX"]) & (df[dof_s].replace(np.nan, 9999) > 1)
    cond_C = (df["time_XX"] < df["F_g_XX"]) & (df[dof_ns].replace(np.nan, 9999) > 1)
    cond_D = (
        (df[dof_nss].replace(np.nan, 9999) >= 2)
        & (
            (((df["F_g_XX"] - 1 + i) == df["time_XX"]) & ((df[dof_s] == 1) & df[dof_s].notnull()))
            | ((df["time_XX"] < df["F_g_XX"]) & ((df[dof_ns] == 1) & (df[dof_ns].notnull())))
        )
    )

    # gen DOF = 1 if A
    df.loc[cond_A, DOF] = 1.0

    # replace with sqrt(dof_s/(dof_s-1)) if B
    idx = cond_B
    vals = df.loc[idx, dof_s]
    df.loc[idx, DOF] = np.sqrt(vals / (vals - 1))
    df.loc[df[dof_s].isna(), DOF] = np.nan


    # replace with sqrt(dof_ns/(dof_ns-1)) if C
    idx = cond_C
    vals = df.loc[idx, dof_ns]
    df.loc[idx, DOF] = np.sqrt(vals / (vals - 1))
    # df.loc[(df[dof_ns].isna() & df[DOF].notnull()), DOF] = np.nan


    # replace with sqrt(dof_nss/(dof_nss-1)) if D
    idx = cond_D
    vals = df.loc[idx, dof_nss]
    df.loc[idx, DOF] = np.sqrt(vals / (vals - 1))
    df.loc[df[dof_nss].isna(), DOF] = np.nan

    return df
   
def _flatten_vars(extra):
    if extra is None:
        return []
    if isinstance(extra, str):
        return [extra]
    flat = []
    for x in extra:
        if isinstance(x, (list, tuple, set)):
            flat.extend(list(x))
        else:
            flat.append(x)
    return flat

def compute_dof_cohort_ns_s(df, i, cluster_col=None, trends_nonparam=None):
    """
    Replicates the Stata block:

      if no cluster:
        by d_sq_XX `trends_nonparam' time_XX:
          gegen dof_cohort_i_ns_s_t_XX = total(dof_ns_s_i_XX) if dof_ns_s_i_XX==1

      else:
        gen cluster_dof_i_ns_s_XX = cluster if dof_ns_s_i_XX==1
        by d_sq_XX `trends_nonparam' time_XX:
          gegen dof_cohort_i_ns_s_t_XX = nunique(cluster_dof_i_ns_s_XX) if !missing(...)

    Result is broadcast to all rows in the corresponding group.
    """
    group_vars = ["d_sq_XX"] + _flatten_vars(trends_nonparam) + ["time_XX"]

    dof_ns_s   = f"dof_ns_s_{i}_XX"
    out_col    = f"dof_cohort_{i}_ns_s_t_XX"

    if cluster_col is None or cluster_col == "":
        if out_col in df.columns:
            df.drop(out_col, axis = 1)
        # --- Count of rows with dof_ns_s == 1 per group, assigned to all rows in that group
        mask = df[dof_ns_s] == 1
        # Build aggregated counts on filtered rows
        agg2 = (df.loc[mask, group_vars + [dof_ns_s]]
              .groupby(group_vars, as_index=False)[dof_ns_s].sum()
              .rename(columns={dof_ns_s: out_col}))
        
        df = df.merge(agg2, on=group_vars, how="left")
        mask = df[dof_ns_s] == 1
        df.loc[~mask, out_col] = np.nan

    else:
        # --- Unique cluster count among rows with dof_ns_s == 1, broadcast to group
        clust_dof = f"cluster_dof_{i}_ns_s_XX"
        df[clust_dof] = np.where(df[dof_ns_s] == 1, df[cluster_col], np.nan)

        mask = df[clust_dof].notna()
        agg = (
            df.loc[mask, group_vars + [clust_dof]]
              .groupby(group_vars, as_index=False)[clust_dof].nunique()
              .rename(columns={clust_dof: out_col})
        )
        
        df = df.merge(agg, on=group_vars, how="left")
        mask = df[clust_dof].notna()
        df.loc[~mask, out_col] = np.nan
    return df

def compute_ns_s_means_with_nans(df, i, trends_nonparam=None):
  """
  Replicates:
    gen dof_ns_s_i = (dof_s_i==1 | dof_ns_i==1)
    by d_sq trends_nonparam time: gegen count = total(N_gt) if dof_ns_s_i==1
    by d_sq trends_nonparam time: gegen total = total(diff_y_i_N_gt) if dof_ns_s_i==1
    gen mean = total / count
  And: if any constructing var is NaN, keep the output NaN.
  """
  # Names
  dof_s        = f"dof_s_{i}_XX"
  dof_ns       = f"dof_ns_{i}_XX"
  dof_ns_s     = f"dof_ns_s_{i}_XX"
  count_col    = f"count_cohort_{i}_ns_s_t_XX"
  total_col    = f"total_cohort_{i}_ns_s_t_XX"
  mean_col     = f"mean_cohort_{i}_ns_s_t_XX"
  diff_y_col   = f"diff_y_{i}_N_gt_XX"

  # Group keys
  group_vars = ["d_sq_XX"] + _flatten_vars(trends_nonparam) + ["time_XX"]

  # --- Column names
  dof_s = f"dof_s_{i}_XX"
  dof_ns = f"dof_ns_{i}_XX"
  dof_ns_s = f"dof_ns_s_{i}_XX"
  count_col = f"count_cohort_{i}_ns_s_t_XX"
  total_col = f"total_cohort_{i}_ns_s_t_XX"
  mean_col = f"mean_cohort_{i}_ns_s_t_XX"
  dof_cohort = f"dof_cohort_{i}_ns_s_t_XX"
  diff_y = f"diff_y_{i}_N_gt_XX"

  # 1. dof_ns_s = (dof_s==1 or dof_ns==1)
  nontull = df[dof_s].notnull() | df[dof_ns].notnull()
  df[dof_ns_s] = ((df[dof_s] == 1) | (df[dof_ns] == 1)).astype(int)

  # 2. Mean’s denominator
  mask = df[dof_ns_s] == 1
  df[count_col] = (
      df.loc[mask]
      .groupby(group_vars)["N_gt_XX"]
      .transform("sum")
  )
  df.loc[~nontull, count_col] = np.nan

  # 3. Mean’s numerator
  df[total_col] = (
      df.loc[mask]
      .groupby(group_vars)[diff_y]
      .transform("sum")
  )
  df.loc[~nontull, total_col] = np.nan
  # 4. Mean
  df[mean_col] = df[total_col] / df[count_col]
  return df


def did_multiplegt_dyn_core(
    df,
    outcome,
    group,
    time,
    treatment, 
    effects, 
    placebo,
    trends_nonparam,
    controls,
    normalized,
    same_switchers, 
    same_switchers_pl, 
    only_never_switchers,
    globals_dict,
    dict_glob,
    const,
    controls_globals,
    trends_lin,
    less_conservative_se,
    continuous,
    cluster,
    switchers_core = None, 
    **kwargs
):
    
    # CRAN Compliance
    list_names_const = []
    F_g_XX = None
    N_gt_XX = None
    T_d_XX = None
    cum_fillin_XX = None
    d_fg_XX_temp = None
    d_sq_int_XX = None
    dum_fillin_temp_XX = None
    dum_fillin_temp_pl_XX = None
    dummy_XX = None
    fillin_g_XX= None
    fillin_g_pl_XX = None
    group_XX = None
    num_g_paths_0_XX = None
    outcome_XX = None
    path_0_XX= None
    relevant_y_missing_XX = None
    sum_temp_XX = None
    sum_temp_pl_XX = None
    time_XX = None
    
    import numpy as np
    dict_vars_gen = {}
    # Inherited Globals
    L_u_XX = globals_dict.get("L_u_XX", np.nan)
    L_placebo_u_XX = globals_dict.get("L_placebo_u_XX", None)
    L_placebo_a_XX = globals_dict.get("L_placebo_a_XX", None)
    L_a_XX = globals_dict.get("L_a_XX", None)
    t_min_XX = globals_dict.get("t_min_XX", None)
    T_max_XX = globals_dict.get("T_max_XX", None)
    G_XX = globals_dict.get("G_XX", None)


    # Assuming df is a pandas DataFrame
    if switchers_core == "in":
        l_u_a_XX = np.nanmin( np.array( [ L_u_XX, effects ] ) )
        if placebo != 0:
            l_placebo_u_a_XX = np.nanmin( np.array( [ placebo, L_placebo_u_XX ] ) )
        increase_XX = 1

    elif switchers_core == "out":
        l_u_a_XX = np.nanmin( np.array( [ L_a_XX, effects ] ) )
        if placebo != 0:
            l_placebo_u_a_XX = np.nanmin( np.array( [ placebo, L_placebo_a_XX ] ) )
        increase_XX = 0

    # Initializing values of baseline treatment
    levels_d_sq_XX = df['d_sq_int_XX'].astype('category').cat.categories.tolist()

    # Remove columns safely
    df.drop(columns=['num_g_paths_0_XX', 'cohort_fullpath_0_XX'], inplace=True, errors='ignore')

    if cluster is None:
        print("No cluster")
    print(f"{int(l_u_a_XX +1)} Number of effects")
    for i in range(1, int(l_u_a_XX +1)):

        cols_to_drop = [
            f"distance_to_switch_{i}_XX",
            f"never_change_d_{i}_XX",
            f"N{increase_XX}_t_{i}_XX",
            f"N{increase_XX}_t_{i}_XX_w",
            f"N{increase_XX}_t_{i}_g_XX",
            f"N_gt_control_{i}_XX",
            f"diff_y_{i}_XX",
            f"dummy_U_Gg{i}_XX",
            f"U_Gg{i}_temp_XX",
            f"U_Gg{i}_XX",
            f"count{i}_core_XX",
            f"U_Gg{i}_temp_var_XX",
            f"U_Gg{i}_var_XX",
            f"never_change_d_{i}_wXX",
            f"distance_to_switch_{i}_wXX",
            f"d_fg{i}_XX",
            f"path_{i}_XX",
            f"num_g_paths_{i}_XX",
            f"cohort_fullpath_{i}_XX",
            f"count_cohort_{i}_s_t_XX",
            f"dof_cohort_{i}_s_t_XX",
            f"dof_cohort_{i}_ns_s_t_XX",
            f"dof_cohort_{i}_ns_t_XX",
            f"dof_cohort_{i}_s0_t_XX",
            f"dof_cohort_{i}_s1_t_XX",
            f"dof_cohort_{i}_s2_t_XX",
            f"dof_cohort_{i}_ns_s_t_XX"
            f"count_cohort_{i}_s_t_XX",
            f"count_cohort_{i}_ns_t_XX",
            f"count_cohort_{i}_s0_t_XX",
            f"count_cohort_{i}_s1_t_XX",
            f"count_cohort_{i}_s2_t_XX",
            f"total_cohort_{i}_s_t_XX",
            f"total_cohort_{i}_ns_t_XX",
            f"total_cohort_{i}_s0_t_XX",
            f"total_cohort_{i}_s1_t_XX",
            f"total_cohort_{i}_s2_t_XX",
            f"mean_cohort_{i}_s_t_XX",
            f"mean_cohort_{i}_ns_t_XX",
            f"mean_cohort_{i}_s0_t_XX",
            f"mean_cohort_{i}_s1_t_XX",
            f"mean_cohort_{i}_s2_t_XX",
            f"E_hat_gt_{i}_XX",
            f"DOF_gt_{i}_XX"
        ]

        # Drop columns if they exist (like Stata’s `capture drop`)
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)


        # 2. sort for the lag calculation
        df.sort_values(["group_XX", "time_XX"], inplace=True)

        # 3. compute long‐difference: outcome_XX - lag(outcome_XX, i)
        df[f"diff_y_{i}_XX"] = (
            df.groupby("group_XX")["outcome_XX"]
            .diff(periods=i)
        )
        # inside your loop over i:
        if less_conservative_se:
            # 1. temporary flag where time == F_g + i - 1
            df['d_fg_XX_temp'] = np.where(
                df['time_XX'] == df['F_g_XX'] + i - 1,
                df['treatment_XX'],
                np.nan
            )

            # 2. group‐level mean of that temp → d_fg{i}_XX
            df[f'd_fg{i}_XX'] = (
                df.groupby('group_XX')['d_fg_XX_temp']
                .transform(lambda x: x.mean(skipna=True))
            )

            # 3. when i == 1, initialize d_fg0_XX & path_0_XX
            if i == 1:
                df['d_fg0_XX'] = df['d_sq_XX']
                # categorical cohort id for (d_fg0_XX, F_g_XX)
                df['path_0_XX'] = (
                    pd.factorize(
                        list(zip(df['d_fg0_XX'], df['F_g_XX']))
                    )[0] + 1
                )

            # 4. carry forward missing d_fg{i} from the previous period
            if i > 1:
                df[f'd_fg{i}_XX'] = df[f'd_fg{i}_XX'].fillna(df[f'd_fg{i-1}_XX'])

            # 5. build the new path_i grouping on (path_{i-1}, d_fg{i})
            df[f'path_{i}_XX'] = (
                pd.factorize(
                    list(zip(
                        df[f'path_{i-1}_XX'],
                        df[f'd_fg{i}_XX']
                    ))
                )[0] + 1
            )

            # 6. drop the temp column
            df.drop(columns='d_fg_XX_temp', inplace=True)

            # 7. for i == 1, count and flag cohorts on path_0_XX
            if i == 1:
                df['num_g_paths_0_XX'] = (
                    df.groupby('path_0_XX')['group_XX']
                    .transform('nunique')
                )
                df['cohort_fullpath_0_XX'] = (df['num_g_paths_0_XX'] > 1).astype(int)

            # 8. for every i, count distinct groups per path_i and flag
            df[f'num_g_paths_{i}_XX'] = (
                df.groupby(f'path_{i}_XX')['group_XX']
                .transform('nunique')
            )
            df[f'cohort_fullpath_{i}_XX'] = (
                df[f'num_g_paths_{i}_XX'] > 1
            ).astype(int)

        # Column names
        never_change = f"never_change_d_{i}_XX"
        never_change_w = f"never_change_d_{i}_wXX"
        N_gt_control = f"N_gt_control_{i}_XX"
        diff_y = f"diff_y_{i}_XX"

        # 1. Create never_change_d_i_XX
        df[never_change] = np.nan
        idx = df[diff_y].notna()
        df.loc[ idx, never_change ] = ( df.loc[idx, "F_g_XX"] > df.loc[ idx, "time_XX"] ) * 1
        # df[never_change] = np.where(
        #      & , 1, 0
        # )

        # 2. Adjust if only_never_switchers option is on
        if only_never_switchers:
            condition = (
                (df["F_g_XX"] > df["time_XX"])
                & (df["F_g_XX"] < T_max_XX + 1)
                & df[diff_y].notna()
            )
            df.loc[condition, never_change] = 0

        # 3. Weighted never_change
        df[never_change_w] = df[never_change] * df["N_gt_XX"]

        # 4. Sum within (time, d_sq, trends_nonparam)
        group_vars = ["time_XX", "d_sq_XX"]
        if trends_nonparam is not None:
            group_vars += trends_nonparam
        df[N_gt_control] = (
            df.groupby(group_vars)[never_change_w]
            .transform("sum")
        )

        # Ensure df is sorted
        df = df.sort_values(['group_XX', 'time_XX'])

        if same_switchers:

            # ==== Step 1: init ====
            df = df.sort_values(["group_XX", "time_XX"])
            df["N_g_control_check_XX"] = 0

            # ==== Step 2: loop over j ====
            for j in range(1, effects + 1):

                # --- long difference ---
                df[f"diff_y_last_XX_{j}"] = (
                    df.groupby("group_XX")["outcome_XX"].diff(j)
                )

                # --- never change indicator ---
                df[f"never_change_d_last_XX_{j}"] = np.where(
                    (~df[f"diff_y_last_XX_{j}"].isna()) & (df["F_g_XX"] > df["time_XX"]),
                    1, 0
                )

                # --- adjust if only_never_switchers flag is active ---
                if only_never_switchers:  
                    cond = (
                        (df["F_g_XX"] > df["time_XX"]) &
                        (df["F_g_XX"] < df["T_max_XX"] + 1) &
                        (~df[f"diff_y_last_XX_{j}"].isna())
                    )
                    df.loc[cond, f"never_change_d_last_XX_{j}"] = 0

                # --- weighted version ---
                df[f"never_change_d_last_wXX_{j}"] = (
                    df[f"never_change_d_last_XX_{j}"] * df["N_gt_XX"]
                )

                # --- group totals (gegen total) ---
                group_cols = ["time_XX", "d_sq_XX"]
                if trends_nonparam:
                    group_cols.append(trends_nonparam)

                df[f"N_gt_control_last_XX_{j}"] = df.groupby(group_cols)[
                    f"never_change_d_last_wXX_{j}"
                ].transform("sum")

                # --- group mean at cutoff time ---
                df[f"N_g_control_last_temp_XX_{j}"] = np.where(
                    df["time_XX"] == df["F_g_XX"] - 1 + j,
                    df[f"N_gt_control_last_XX_{j}"],
                    np.nan
                )
                df[f"N_g_control_last_m_XX_{j}"] = df.groupby("group_XX")[
                    f"N_g_control_last_temp_XX_{j}"
                ].transform("mean")

                # --- relevant diff_y at cutoff ---
                df[f"diff_y_relev_temp_XX_{j}"] = np.where(
                    df["time_XX"] == df["F_g_XX"] - 1 + j,
                    df[f"diff_y_last_XX_{j}"],
                    np.nan
                )
                df[f"diff_y_relev_XX_{j}"] = df.groupby("group_XX")[
                    f"diff_y_relev_temp_XX_{j}"
                ].transform("mean")

                # --- update check counter ---
                df["N_g_control_check_XX"] += (
                    (df[f"N_g_control_last_m_XX_{j}"] > 0) &
                    (~df[f"diff_y_relev_XX_{j}"].isna())
                ).astype(int)

            if same_switchers_pl:  # equivalent to if we are using true or false for this variable ("`same_switchers_pl'" != "")

                # Make sure panel is ordered like xtset group_XX time_XX
                df = df.sort_values(["group_XX", "time_XX"]).copy()

                # N_g_control_check_pl_XX = 0
                df["N_g_control_check_pl_XX"] = 0

                if trends_nonparam is None:
                    group_cols = ["time_XX", "d_sq_XX"]
                else:
                    group_cols = ["time_XX", "d_sq_XX"] + trends_nonparam

                # forv j = 1/`placebo'
                for j in range(1, int(placebo + 1)):

                    # diff_y_last_XX = outcome_XX - F`j'.outcome_XX  (lead j)
                    df["diff_y_last_XX"] = (
                        df.groupby("group_XX")["outcome_XX"]
                        .transform(lambda s, j=j: s - s.shift(-j))
                    )

                    # never_change_d_last_XX = (F_g_XX > time_XX) if diff_y_last_XX != .
                    df["never_change_d_last_XX"] = np.where(
                        df["diff_y_last_XX"].notna(),
                        (df["F_g_XX"] > df["time_XX"]),
                        np.nan
                    )

                    if only_never_switchers:
                        mask_never = (
                            (df["F_g_XX"] > df["time_XX"]) &
                            (df["F_g_XX"] < df["T_max_XX"] + 1) &
                            df["diff_y_last_XX"].notna()
                        )
                        df.loc[mask_never, "never_change_d_last_XX"] = 0

                    # never_change_d_last_wXX = never_change_d_last_XX * N_gt_XX
                    df["never_change_d_last_wXX"] = (
                        df["never_change_d_last_XX"] * df["N_gt_XX"]
                    )

                    # bys time_XX d_sq_XX `trends_nonparam': total(never_change_d_last_wXX)
                    df["N_gt_control_last_XX"] = (
                        df.groupby(group_cols)["never_change_d_last_wXX"]
                        .transform("sum")
                    )

                    # N_g_control_last_temp_XX = N_gt_control_last_XX if time_XX == F_g_XX - 1 - j
                    mask_time_j = (df["time_XX"] == df["F_g_XX"] - 1 - j)
                    df["N_g_control_last_temp_XX"] = np.where(
                        mask_time_j,
                        df["N_gt_control_last_XX"],
                        np.nan
                    )

                    # bys group_XX: mean(N_g_control_last_temp_XX)
                    df["N_g_control_last_m_XX"] = (
                        df.groupby("group_XX")["N_g_control_last_temp_XX"]
                        .transform("mean")
                    )

                    # diff_y_relev_temp_XX = diff_y_last_XX if time_XX == F_g_XX - 1 - j
                    df["diff_y_relev_temp_XX"] = np.where(
                        mask_time_j,
                        df["diff_y_last_XX"],
                        np.nan
                    )

                    # bys group_XX: mean(diff_y_relev_temp_XX)
                    df["diff_y_relev_XX"] = (
                        df.groupby("group_XX")["diff_y_relev_temp_XX"]
                        .transform("mean")
                    )

                    # N_g_control_check_pl_XX += (N_g_control_last_m_XX > 0 & diff_y_relev_XX != .)
                    df["N_g_control_check_pl_XX"] = (
                        df["N_g_control_check_pl_XX"] +
                        ((df["N_g_control_last_m_XX"] > 0) &
                        df["diff_y_relev_XX"].notna()).astype(int)
                    )

                # relevant_y_missing_XX = (outcome_XX == . & time in window)
                window_mask = (
                    (df["time_XX"] >= df["F_g_XX"] - 1 - placebo) &
                    (df["time_XX"] <= df["F_g_XX"] - 1 + effects)
                )
                df["relevant_y_missing_XX"] = (
                    df["outcome_XX"].isna() & window_mask
                )

                # If controls option is on:
                if controls:
                    mask_ctrl = (
                        (df["fd_X_all_non_missing_XX"] == 0) &
                        window_mask
                    )
                    df.loc[mask_ctrl, "relevant_y_missing_XX"] = True

                # fillin_g_pl_XX = (N_g_control_check_pl_XX == placebo)
                df["fillin_g_pl_XX"] = (df["N_g_control_check_pl_XX"] == placebo)

                # still_switcher_`i'_XX = (F_g_XX - 1 + effects <= T_g_XX & N_g_control_check_XX == effects)
                still_col = f"still_switcher_{i}_XX"
                df[still_col] = (
                    (df["F_g_XX"] - 1 + effects <= df["T_g_XX"]) &
                    (df["N_g_control_check_XX"] == effects)
                )

                # distance_to_switch_`i'_XX =
                # (still_switcher_i & time_XX == F_g_XX - 1 + i
                #  & i <= L_g_XX & S_g_XX == increase_XX
                #  & N_gt_control_i_XX > 0 & N_gt_control_i_XX != .) if diff_y_i_XX != .
                dist_col = f"distance_to_switch_{i}_XX"
                n_gt_ctrl_col = f"N_gt_control_{i}_XX"
                diff_y_i_col = f"diff_y_{i}_XX"

                mask_distance = (
                    df[still_col] &
                    (df["time_XX"] == df["F_g_XX"] - 1 + i) &
                    (i <= df["L_g_XX"]) &
                    (df["S_g_XX"] == increase_XX) &   # both are columns
                    (df[n_gt_ctrl_col] > 0) &
                    df[n_gt_ctrl_col].notna() &
                    df[diff_y_i_col].notna()
                )

                # Stata gen ... if diff_y_i_XX != .  →  NaN where condition is false
                df[dist_col] = np.where(mask_distance, 1, 0)

            else:

                # ---- relevant_y_missing ----
                df["relevant_y_missing_XX"] = (
                    df["outcome_XX"].isna() &
                    (df["time_XX"] >= df["F_g_XX"] - 1) &
                    (df["time_XX"] <= df["F_g_XX"] - 1 + effects)
                )

                if controls:  # equivalent to if "`controls'" != ""
                    cond = (
                        (df["fd_X_all_non_missing_XX"] == 0) &
                        (df["time_XX"] >= df["F_g_XX"]) &
                        (df["time_XX"] <= df["F_g_XX"] - 1 + effects)
                    )
                    df.loc[cond, "relevant_y_missing_XX"] = 1

                # ---- still_switcher ----
                df[f"still_switcher_{i}_XX"] = np.nan
                df[f"still_switcher_{i}_XX"] = (
                    (df["F_g_XX"] - 1 + effects <= df["T_g_XX"]) &
                    (df["N_g_control_check_XX"] == effects)
                ).astype(int)

                # ---- distance_to_switch ----
                df[f"distance_to_switch_{i}_XX"] = np.nan
                idx = (df[f"diff_y_{i}_XX"].isna())
                df[ f"distance_to_switch_{i}_XX"] = (
                    ( df[f"still_switcher_{i}_XX"] ==1 ) &
                    (df["time_XX"] == df["F_g_XX"] - 1 + i) &
                    (i <= df["L_g_XX"]) &
                    (df["S_g_XX"] == increase_XX) &
                    (df[f"N_gt_control_{i}_XX"] > 0) &
                    (~df[f"N_gt_control_{i}_XX"].isna())
                    ).astype(int)
                df.loc[ idx, f"distance_to_switch_{i}_XX"] = np.nan
        else:
            # Distance to switch: basic version without same_switchers
            col_dist  = f"distance_to_switch_{i}_XX"
            col_diff  = f"diff_y_{i}_XX"
            col_ctrl  = f"N_gt_control_{i}_XX"

            df[col_dist] = np.nan
            idx = df[col_diff].isna()
            df[col_dist] = (
                (df[ "time_XX"] == df["F_g_XX"] - 1 + i) * (i <= df["L_g_XX"]) *  (df["S_g_XX"] == increase_XX) * (df[col_ctrl] > 0) * (~df[col_ctrl].isna()))
            df.loc[idx, col_dist] = np.nan
            
        # Necesitamos checkear como se estan contando los switchers
        # Ensure the "distance_to_switch" column is numeric
        df[f"distance_to_switch_{i}_XX"] = df[f"distance_to_switch_{i}_XX"].astype(float)

        # Create weighted distance variable
        df[f"distance_to_switch_{i}_wXX"] = df[f"distance_to_switch_{i}_XX"] * df["N_gt_XX"]

        # Sum over time to get counts per period
        df[f"N{increase_XX}_t_{i}_XX"] = (
            df.groupby("time_XX")[f"distance_to_switch_{i}_wXX"]
            .transform("sum")
        )
        df[f"N_dw{increase_XX}_t_{i}_XX"] = (
            df.groupby("time_XX")[f"distance_to_switch_{i}_XX"]
            .transform("sum")
        )


        dict_vars_gen[f"N{increase_XX}_{i}_XX"] = 0
        dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"] = 0
        n_placebo = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
        n_dw_placebo = dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"]
        # ——— Loop over time and add up the period‐means ———
        for t in range(int(t_min_XX),int( T_max_XX + 1)):
            # build your column names
            col_p  = f"N{increase_XX}_t_{i}_XX"
            col_dp = f"N_dw{increase_XX}_t_{i}_XX"
            
            # mask for this period
            mask = df['time_XX'] == t
            
            # add the mean of each column (skipna=True drops NaNs)
            n_placebo    += df.loc[mask, col_p ].mean(skipna=True)
            n_dw_placebo += df.loc[mask, col_dp].mean(skipna=True)

        # ——— (Optional) Put them in a dict if you want “dynamic” names ———
        dict_vars_gen[f"N{increase_XX}_{i}_XX"] = n_placebo
        dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"] = n_dw_placebo

        # Count groups ℓ periods away from switch by (time, baseline treatment, trends)
        
        if trends_nonparam is None:
            group_cols = ["time_XX", "d_sq_XX"]
        else:
            group_cols = ["time_XX", "d_sq_XX"] + trends_nonparam
        df[f"N{increase_XX}_t_{i}_g_XX"] = (
            df.groupby(group_cols)[f"distance_to_switch_{i}_wXX"]
            .transform("sum")
        )
        if controls:
            # Initialize intermediate variable
            df[f"part2_switch{increase_XX}_{i}_XX"] = 0

            # Compute T_d_XX: last period by baseline treatment
            df["T_d_XX"] = df.groupby("d_sq_int_XX")["F_g_XX"].transform("max") - 1

            count_controls = 0
            for var in controls:
                count_controls += 1

                # Sort by panel identifiers
                df = df.sort_values(["group_XX", "time_XX"])

                # Compute lags within groups
                df[f"L{i}_{var}"] = df.groupby("group_XX")[var].shift(i)

                # Compute the placebo difference
                diff_col = f"diff_X{count_controls}_{i}_XX"
                df[diff_col] = df[f"{var}"] - df[f"L{i}_{var}"]
                
                # Weighted long difference
                diff_n_col = f"diff_X{count_controls}_{i}_N_XX"
                df[diff_n_col] = df["N_gt_XX"] * df[diff_col]

                def safe_div(num, den):
                    den = np.asarray(den)
                    return np.where((den != 0) & ~np.isnan(den), num / den, np.nan)
                df["G_XX"] = G_XX
                for l in levels_d_sq_XX:  # l corresponds to d in the paper
                    l = int(l)
                    # ----- m^{+/-}_{g,d,l} inside-summation term -----
                    m_g_col   = f"m{increase_XX}_g_{count_controls}_{l}_{i}_XX"
                    m_sum_col = f"m{increase_XX}_{l}_{count_controls}_{i}_XX"
                    M_col     = f"M{increase_XX}_{l}_{count_controls}_{i}_XX"

                    # Conditions
                    cond1 = (i <= (df["T_g_XX"] - 2)) & (df["d_sq_int_XX"] == l)
                    cond2 = (df["time_XX"] >= (i + 1)) & (df["time_XX"] <= df["T_g_XX"])

                    # Inner pieces
                    num_inner = (
                        df[f"distance_to_switch_{i}_XX"]
                        - safe_div(df[f"N{increase_XX}_t_{i}_g_XX"], df[f"N_gt_control_{i}_XX"])
                        * df[f"never_change_d_{i}_XX"]
                    )
                    df[f"N{increase_XX}_{i}_XX"] = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
                    frac = safe_div(df["G_XX"], df[f"N{increase_XX}_{i}_XX"])

                    # m`increase'_g_count_l_i_XX
                    df[m_g_col] = (
                        cond1.astype(float)
                        * frac
                        * (num_inner * cond2.astype(float) * df[f"diff_X{count_controls}_{i}_N_XX"])
                    )

                    # Sum across t within group g, keep only first row per group (others -> NaN)
                    df[m_sum_col] = df.groupby("group_XX")[m_g_col].transform("sum")
                    first_in_group = df.groupby("group_XX").cumcount() == 0
                    df.loc[~first_in_group, m_sum_col] = np.nan

                    # ----- M^{+/-}_{d,l} : total of m over all obs, scaled by 1/G_XX -----
                    total_m = np.nansum(df[m_sum_col].to_numpy())
                    df[M_col] = safe_div(total_m, df["G_XX"])

                    # ----- Number of groups within each not-yet-switched cohort (E_hat_denom...) -----
                    # dummy_XX = (F_g_XX > time_XX) & (d_sq_int_XX == l) if diff_y_XX is not missing
                    df["dummy_XX"] = 0
                    df.loc[(df["diff_y_XX"].notna()) & (df["F_g_XX"] > df["time_XX"]) & (df["d_sq_int_XX"] == l), "dummy_XX"] = 1

                    E_hat_col = f"E_hat_denom_{count_controls}_{l}_XX"
                    if cluster is None:  # no clustering: count groups (sum of dummy) by time_XX
                        counts_by_time = df.groupby("time_XX")["dummy_XX"].sum(min_count=1)
                        df[E_hat_col] = df["time_XX"].map(counts_by_time)
                        # Keep value only where d_sq_int==l; else NaN (mimic Stata if-condition)
                        df.loc[df["d_sq_int_XX"] != l, E_hat_col] = np.nan
                    else:
                        # clustering: nunique over cluster among dummy==1
                        df["cluster_temp_XX"] = np.where(df["dummy_XX"] == 1, df[cluster], np.nan)
                        nunique_by_time = df.groupby("time_XX")["cluster_temp_XX"].nunique(dropna=True)
                        df[E_hat_col] = df["time_XX"].map(nunique_by_time)
                        # Only define where cluster_temp exists; else NaN
                        df.loc[df["cluster_temp_XX"].isna(), E_hat_col] = np.nan

                    # ----- Indicator for at least two groups in cohort (demeaning possible) -----
                    Ey_hat_gt_col = f"E_y_hat_gt_{l}_XX"             # target
                    Ey_hat_gt_int_col = f"E_y_hat_gt_int_{l}_XX"     # assumed precomputed column
                    df[Ey_hat_gt_col] = df[Ey_hat_gt_int_col] * (df[E_hat_col] >= 2)

                    # ----- Summation from t=2 to F_g-1 in U^{+,var,X}_{g,l} / U^{-,var,X}_{g,l} -----
                    in_sum_temp_col     = f"in_sum_temp_{count_controls}_{l}_XX"
                    in_sum_temp_adj_col = f"in_sum_temp_adj_{count_controls}_{l}_XX"
                    in_sum_col          = f"in_sum_{count_controls}_{l}_XX"

                    # N_c_l_temp_XX and N_c_l_XX
                    N_c_temp_col = f"N_c_{l}_temp_XX"
                    N_c_col      = f"N_c_{l}_XX"

                    # N_c_l_temp_XX = N_gt_XX * 1{ d_sq_int==l, 2<=time<=T_d, time<F_g, diff_y not missing }
                    cond_Nc = (
                        (df["d_sq_int_XX"] == l)
                        & (df["time_XX"] >= 2)
                        & (df["time_XX"] <= df["T_d_XX"])
                        & (df["time_XX"] < df["F_g_XX"])
                        & (df["diff_y_XX"].notna())
                    )
                    df[N_c_temp_col] = np.where(cond_Nc, df["N_gt_XX"], 0.0)

                    # Total across all obs (scalar replicated)
                    total_Nc = float(np.nansum(df[N_c_temp_col].to_numpy()))
                    df[N_c_col] = total_Nc

                    # Adjust demeaning when E_hat_denom == 1
                    df[in_sum_temp_adj_col] = np.where(df[f"{Ey_hat_gt_col}"].notna(), 0.0, np.nan)
                    mask_adj = df[f"{Ey_hat_gt_col}"].notna() & (df[E_hat_col] > 1)
                    df.loc[mask_adj, in_sum_temp_adj_col] = np.sqrt(
                        df.loc[mask_adj, E_hat_col] / (df.loc[mask_adj, E_hat_col] - 1.0)
                    ) - 1.0

                    # Build in-summand:
                    # (prod_X{count}_Ngt_XX * (1 + 1{E_hat>=2} * adj) * (diff_y_XX - E_y_hat_gt_l_XX) * 1{2<=time<=F_g-1}) / N_c_l_XX
                    prod_col = f"prod_X{count_controls}_Ngt_XX"  # assumed existing
                    summand = (
                        df[prod_col]
                        * (1.0 + (df[E_hat_col] >= 2).astype(float) * df[in_sum_temp_adj_col])
                        * (df["diff_y_XX"] - df[Ey_hat_gt_col])
                        * ((df["time_XX"] >= 2) & (df["time_XX"] <= (df["F_g_XX"] - 1))).astype(float)
                    )
                    df[in_sum_temp_col] = safe_div(summand, df[N_c_col])

                    # Sum within group g across t
                    df[in_sum_col] = df.groupby("group_XX")[in_sum_temp_col].transform("sum")
                    
                for l in levels_d_sq_XX:  # assuming this is a Python list of levels
                    l = int(l)
                    if dict_glob[f"useful_res_{l}_XX"] > 1:  # assuming useful_res is a dict or Series of scalars
                        
                        coef = dict_glob[f'coefs_sq_{l}_XX'][count_controls - 1]
                        
                        # careful: in Stata indices start at 1, in pandas .iloc is 0-based
                        mask = df["d_sq_int_XX"] == l
                        df.loc[mask, f"diff_y_{i}_XX"] = (
                            df.loc[mask, f"diff_y_{i}_XX"]
                            - coef * df.loc[mask, f"diff_X{count_controls}_{i}_XX"]
                        )

                        # Drop if exists and then generate new column = 0
                        col_name = f"in_brackets_{l}_{count_controls}_XX"
                        if col_name in df.columns:
                            df.drop(columns=col_name, inplace=True)
                        df[col_name] = 0



        # assuming i and increase_XX are defined, and trends_nonparam is a list of column names

        # 1. Weighted long difference of outcome
        df[f"diff_y_{i}_N_gt_XX"] = df[f"diff_y_{i}_XX"] * df["N_gt_XX"]

        # 2. DOF indicator: 1 if N_gt_XX != 0 and diff_y not missing
        df[f"dof_y_{i}_N_gt_XX"] = (
            (df["N_gt_XX"] != 0) & df[f"diff_y_{i}_XX"].notna()
        ).astype(int)


        # Drop old columns if they exist
        for col in [f"dof_ns_{i}_XX", f"dof_s_{i}_XX"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Generate dof_ns_i_XX
        df[f"dof_ns_{i}_XX"] = (
            (df["N_gt_XX"] != 0) &
            (df[f"diff_y_{i}_XX"].notna()) &
            (df[f"never_change_d_{i}_XX"] == 1) &
            (df[f"N{increase_XX}_t_{i}_XX"] > 0) &
            (df[f"N{increase_XX}_t_{i}_XX"].notna())
        ).astype(int)   # cast to int (0/1) if needed

        # Generate dof_s_i_XX
        mask = (df["N_gt_XX"] != 0) & (df[f"distance_to_switch_{i}_XX"] == 1)

        # If any inputs are NaN → result = NaN
        df[f"dof_s_{i}_XX"] = (( df["N_gt_XX"] != 0 ) & ( df[f"distance_to_switch_{i}_XX"] == 1 )).astype(int)



        # 3. Define mask for "never switchers" with valid diff_y and controls
        mask = ( df[f"dof_ns_{i}_XX"] == 1 )
        
        if trends_nonparam is None:
            group_cols = ["d_sq_XX", "time_XX"]
        else:
            group_cols = ["d_sq_XX"] + trends_nonparam + ["time_XX"]

        # 4. Denominator of the mean: count of control groups weighted by N_gt_XX
        df["_mask_den"] = mask * df["N_gt_XX"]
        df[f"count_cohort_{i}_ns_t_XX"] = (
            df.groupby(group_cols)["_mask_den"]
            .transform("sum")
        )
        df.loc[ ~mask, f"count_cohort_{i}_ns_t_XX"] = np.nan
        df.drop(columns="_mask_den", inplace=True)

        # 5. Numerator of the mean: sum of weighted diff_y over the same mask
        mask = ( df[f"dof_ns_{i}_XX"] == 1 )
        df["_val_num"] = mask * df[f"diff_y_{i}_N_gt_XX"]
        df[f"total_cohort_{i}_ns_t_XX"] = (
            df.groupby(group_cols)["_val_num"]
            .transform("sum")
        )
        df.loc[ ~mask, f"total_cohort_{i}_ns_t_XX"] = np.nan

        # 6. Mean for never-switcher cohort
        df[f"mean_cohort_{i}_ns_t_XX"] = (
            df[f"total_cohort_{i}_ns_t_XX"] /
            df[f"count_cohort_{i}_ns_t_XX"]
        )

        # # 7. DOF for cohort adjustment: count of valid dof observations
        if trends_nonparam is None:
            group_cols = ["d_sq_XX", "time_XX"]
        else:
            group_cols = ["d_sq_XX", "time_XX"] + trends_nonparam

        if cluster is None or cluster == "":  # case: no cluster
            mask = ( df[ f"dof_ns_{i}_XX" ] == 1 ) 
            dfaux = df.loc[mask, :].groupby(group_cols)[f"dof_ns_{i}_XX"].sum().reset_index() \
                                              .rename(columns = { f"dof_ns_{i}_XX" : f"dof_cohort_{i}_ns_t_XX" } )
            df = df.merge(dfaux, on = group_cols, how = 'left')
            mask = ( df[ f"dof_ns_{i}_XX" ] == 1 )
            df.loc[~mask, f"dof_cohort_{i}_ns_t_XX"] = np.nan

        else:  # case: cluster is provided
            
            cluster_col = cluster
            if trends_nonparam is None:
                group_vars = ["d_sq_XX"] + ["time_XX"]
            else:
                group_vars = ["d_sq_XX"] + trends_nonparam + ["time_XX"]

            # 1) cluster_dof_i_ns_XX = cluster when dof_ns_i_XX == 1, else NaN
            dof_ns = f"dof_ns_{i}_XX"
            clust_dof = f"cluster_dof_{i}_ns_XX"
            df[clust_dof] = np.where(df[dof_ns] == 1, df[cluster_col], np.nan)

            # 2) Grouping vars: d_sq_XX, trends_nonparam..., time_XX
           

            # 3) Count unique clusters within groups, only where cluster_dof not missing
            out_col = f"dof_cohort_{i}_ns_t_XX"
            mask = df[clust_dof].notna()

            counts = (
                df.loc[mask]
                .groupby(group_vars)[clust_dof]
                .transform("nunique")
            )

            # Assign only to rows where mask is True (others remain NaN, matching Stata's "if !missing(...)")
            df.loc[mask, out_col] = counts




        # Assume i, less_conservative_se, trends_nonparam are defined,
        # and df is your pandas DataFrame

        if not less_conservative_se:

            # Group variables
            # group_vars = ["d_sq_XX", "F_g_XX", "d_fg_XX"]
            # if trends_nonparam is not None:
            if trends_nonparam is None:
                print('trends_nonparam is None')
                group_vars = ['d_sq_XX', 'F_g_XX', 'd_fg_XX', f'distance_to_switch_{i}_XX']
            else:
                group_vars = ['d_sq_XX', 'F_g_XX', 'd_fg_XX', f'distance_to_switch_{i}_XX'] + trends_nonparam

            # Column names
            dof_s = f"dof_s_{i}_XX"
            count_col = f"count_cohort_{i}_s_t_XX"
            total_col = f"total_cohort_{i}_s_t_XX"
            mean_col = f"mean_cohort_{i}_s_t_XX"
            dof_cohort = f"dof_cohort_{i}_s_t_XX"
            diff_y = f"diff_y_{i}_N_gt_XX"

            # cambio1
            mask = df[dof_s] == 1

            # --- Mean's denominator: sum of N_gt_XX over rows with mask, broadcast to all rows in the group
            selcols = group_vars + ["N_gt_XX"]
            den = (
                df.loc[mask, selcols]
                .groupby(group_vars, as_index=False)["N_gt_XX"].sum()
                .rename(columns={"N_gt_XX": count_col})
            )
            df = df.merge(den, on=group_vars, how="left")
            mask = df[dof_s] == 1

            # --- Mean's numerator: sum of diff_y over rows with mask, broadcast to all rows in the group
            selcols = group_vars + [diff_y]
            num = (
                df.loc[mask, selcols]
                .groupby(group_vars, as_index=False)[diff_y].sum()
                .rename(columns={diff_y: total_col})
            )
            df = df.merge(num, on=group_vars, how="left")

            # --- Mean (numerator / denominator) ---
            df[mean_col] = df[total_col] / df[count_col]

            # --- Degrees of freedom adjustment ---
            cluster_col = cluster  # assuming 'cluster' is defined elsewhere
            if cluster_col is None or cluster_col == "":
                # Just sum dof_s
                df[dof_cohort] = (
                    df.loc[df[dof_s] == 1]
                    .groupby(group_vars)[dof_s]
                    .transform("sum")
                )
            else:
                # Use unique clusters
                cluster_dof = f"cluster_dof_{i}_s_XX"
                df[cluster_dof] = np.where(df[dof_s] == 1, df[cluster_col], np.nan)

                df[dof_cohort] = (
                    df.loc[~df[cluster_dof].isna()]
                    .groupby(group_vars)[cluster_dof]
                    .transform("nunique")
                )

        else:

            def apply_less_conservative_se(
                df: pd.DataFrame,
                i: int,
                trends_nonparam=None,
                less_conservative_se: bool = True,
                cluster_col: str | None = None,
            ):
                if not less_conservative_se:
                    return df

                if trends_nonparam is None:
                    trends_nonparam = []

                df = df.copy()

                dof_s_col = f"dof_s_{i}_XX"
                diff_yN_col = f"diff_y_{i}_N_gt_XX"

                cond_dof = df[dof_s_col] == 1

                # Helper to build group keys list
                def group_keys(path_col):
                    return [df[path_col]] + [df[c] for c in trends_nonparam]

                # --------- Mean's denominator: count_cohort_* (sum of N_gt_XX) ---------
                for path_col, s_tag in [
                    ("path_0_XX", "s0"),
                    ("path_1_XX", "s1"),
                    (f"path_{i}_XX", "s2"),
                ]:
                    tmp = df["N_gt_XX"].where(cond_dof)
                    col = f"count_cohort_{i}_{s_tag}_t_XX"

                    df[col] = tmp.groupby(group_keys(path_col)).transform("sum")
                    # Stata: gegen ... if dof_s == 1 → missing when condition is false
                    df.loc[~cond_dof, col] = np.nan

                # --------- Mean's numerator: total_cohort_* (sum of diff_y_i_N_gt_XX) ---------
                for path_col, s_tag in [
                    ("path_0_XX", "s0"),
                    ("path_1_XX", "s1"),
                    (f"path_{i}_XX", "s2"),
                ]:
                    tmp = df[diff_yN_col].where(cond_dof)
                    col = f"total_cohort_{i}_{s_tag}_t_XX"

                    df[col] = tmp.groupby(group_keys(path_col)).transform("sum")
                    df.loc[~cond_dof, col] = np.nan

                # --------- Counting number of groups for DOF adjustment ---------
                if cluster_col is None:
                    # No clustering: sum of dof_s_i_XX
                    tmp = df[dof_s_col].where(cond_dof)
                    for path_col, s_tag in [
                        ("path_0_XX", "s0"),
                        ("path_1_XX", "s1"),
                        (f"path_{i}_XX", "s2"),
                    ]:
                        col = f"dof_cohort_{i}_{s_tag}_t_XX"
                        df[col] = tmp.groupby(group_keys(path_col)).transform("sum")
                        df.loc[~cond_dof, col] = np.nan
                else:
                    # Clustering: number of unique clusters among rows with dof_s == 1
                    cluster_dof_col = f"cluster_dof_{i}_s_XX"
                    df[cluster_dof_col] = np.where(cond_dof, df[cluster_col], np.nan)

                    tmp = df[cluster_dof_col]
                    for path_col, s_tag in [
                        ("path_0_XX", "s0"),
                        ("path_1_XX", "s1"),
                        (f"path_{i}_XX", "s2"),
                    ]:
                        col = f"dof_cohort_{i}_{s_tag}_t_XX"
                        df[col] = tmp.groupby(group_keys(path_col)).transform("nunique")
                        # Stata: nunique(...) if !missing(cluster_dof) → missing when NA
                        df.loc[tmp.isna(), col] = np.nan

                # --------- Choose which cohort's DoF to use (s2, then s1, then s0) ---------
                col_s0 = f"dof_cohort_{i}_s0_t_XX"
                col_s1 = f"dof_cohort_{i}_s1_t_XX"
                col_s2 = f"dof_cohort_{i}_s2_t_XX"
                col_st = f"dof_cohort_{i}_s_t_XX"

                df[col_st] = np.nan

                cond_s2_ge2 = df[col_s2] >= 2
                df.loc[cond_s2_ge2, col_st] = df.loc[cond_s2_ge2, col_s2]

                cond_s1_ge2 = (df[col_s2] < 2) & (df[col_s1] >= 2)
                df.loc[cond_s1_ge2, col_st] = df.loc[cond_s1_ge2, col_s1]

                cond_else = (df[col_s2] < 2) & (df[col_s1] < 2)
                df.loc[cond_else, col_st] = df.loc[cond_else, col_s0]

                # --------- Mean: pick s2, else s1, else s0 ---------
                col_cnt_s0 = f"count_cohort_{i}_s0_t_XX"
                col_cnt_s1 = f"count_cohort_{i}_s1_t_XX"
                col_cnt_s2 = f"count_cohort_{i}_s2_t_XX"

                col_tot_s0 = f"total_cohort_{i}_s0_t_XX"
                col_tot_s1 = f"total_cohort_{i}_s1_t_XX"
                col_tot_s2 = f"total_cohort_{i}_s2_t_XX"

                col_mean = f"mean_cohort_{i}_s_t_XX"
                df[col_mean] = np.nan

                # s2
                cond_s2_ge2 = df[col_s2] >= 2
                df.loc[cond_s2_ge2, col_mean] = (
                    df.loc[cond_s2_ge2, col_tot_s2] / df.loc[cond_s2_ge2, col_cnt_s2]
                )

                # s1
                cond_s1_ge2 = (df[col_s2] < 2) & (df[col_s1] >= 2)
                df.loc[cond_s1_ge2, col_mean] = (
                    df.loc[cond_s1_ge2, col_tot_s1] / df.loc[cond_s1_ge2, col_cnt_s1]
                )

                # s0
                cond_else = (df[col_s2] < 2) & (df[col_s1] < 2)
                df.loc[cond_else, col_mean] = (
                    df.loc[cond_else, col_tot_s0] / df.loc[cond_else, col_cnt_s0]
                )

                return df

            df = apply_less_conservative_se( df, i = int(i), 
                                           trends_nonparam = trends_nonparam, 
                                           less_conservative_se = less_conservative_se, 
                                           cluster_col = cluster)
            # And DOF for that same fallback
            dof_s2 = df[f'dof_cohort_{i}_s2_t_XX']
            dof_s1 = df[f'dof_cohort_{i}_s1_t_XX']
            dof_s0 = df[f'dof_cohort_{i}_s0_t_XX']

            df[f'dof_cohort_{i}_s_t_XX'] = np.where(
                df[f'cohort_fullpath_{i}_XX'] == 1, dof_s2,
                np.where(df['cohort_fullpath_1_XX'] == 1, dof_s1, dof_s0)
            )


        # Generating Variables for Standard Error Calculation
        df = compute_ns_s_means_with_nans(df, i = i, trends_nonparam  = trends_nonparam)
        df = compute_dof_cohort_ns_s(df, i = i, cluster_col = cluster, trends_nonparam  = trends_nonparam  )
        df = compute_E_hat_gt_with_nans(df, i)
        df = compute_DOF_gt_with_nans(df, i)        


        # 8. Generate U_Gg_i
        N_val = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
        if N_val != 0:

            # Dummy for not-yet-treated groups
            df[f"dummy_U_Gg{i}_XX"] = (i <= (df["T_g_XX"] - 1)).astype(int)

            # ==== Define column names ====
            col_temp     = f"U_Gg{i}_temp_XX"
            col_final    = f"U_Gg{i}_XX"
            col_dummy    = f"dummy_U_Gg{i}_XX"
            col_Ni       = f"N{increase_XX}_{i}_XX"
            col_dist     = f"distance_to_switch_{i}_XX"
            col_Nit_g    = f"N{increase_XX}_t_{i}_g_XX"
            col_Nctrl    = f"N_gt_control_{i}_XX"
            col_never    = f"never_change_d_{i}_XX"
            col_diff     = f"diff_y_{i}_XX"
            col_count    = f"count{i}_core_XX"
            col_temp_var = f"U_Gg{i}_temp_var_XX"
            col_DOF      = f"DOF_gt_{i}_XX"
            col_Ehat     = f"E_hat_gt_{i}_XX"
            df[col_Ni] = N_val
            df["G_XX"] = G_XX
            # ==== Step 1: Generate U_Gg{i}_temp_XX ====
            df[col_temp] = (
                df[col_dummy] *
                (df["G_XX"] / df[col_Ni]) *
                ((df["time_XX"] >= (i + 1)) & (df["time_XX"] <= df["T_g_XX"])) *
                df["N_gt_XX"] *
                (
                    df[col_dist] -
                    (df[col_Nit_g] / df[col_Nctrl]) * df[col_never]
                )
            )

            # ==== Step 2: Multiply by diff_y ====
            df[col_temp] = df[col_temp] * df[col_diff]

            # ==== Step 3: Group total (gegen … total) ====
            df[col_final] = df.groupby("group_XX")[col_temp].transform("sum")

            # ==== Step 4: Multiply by first_obs_by_gp_XX ====
            df[col_final] = ( df[col_final] * df["first_obs_by_gp_XX"] ).copy()

            # ==== Step 5: Count core ====
            df[col_count] = 0
            condition = (
                (~df[col_temp].isna() & (df[col_temp] != 0)) |
                (
                    (df[col_temp] == 0) & (df[col_diff] == 0) &
                    (
                        (df[col_dist] != 0) |
                        ((df[col_Nit_g] != 0) & (df[col_never] != 0))
                    )
                )
            )
            df.loc[condition, col_count] = df["N_gt_XX"]

            # ==== Step 6: Init temp_var ====
            df[col_temp_var] = 0

            # ==== Step 7: Final computation ====
            df[col_temp_var] = (
                df[col_dummy] *
                (df["G_XX"] / df[col_Ni]) *
                (
                    df[col_dist] -
                    (df[col_Nit_g] / df[col_Nctrl]) * df[col_never]
                ) *
                ((df["time_XX"] >= (i + 1)) & (df["time_XX"] <= df["T_g_XX"])) *
                df["N_gt_XX"] *
                df[col_DOF] *
                (df[col_diff] - df[col_Ehat])
            )
            # 6. Adjustment for controls, if any
            if controls:
                part2 = f"part2_switch{increase_XX}_{i}_XX"
                df[part2] = 0.0
                for l in levels_d_sq_XX:  # loop over levels
                    l = int(l)
                    if dict_glob[f'useful_res_{l}_XX'] > 1:

                        # Initialize combined temp
                        col_combined_temp = f"combined{increase_XX}_temp_{l}_{i}_XX"
                        if col_combined_temp in df.columns:
                            df.drop(columns=col_combined_temp, inplace=True)
                        df[col_combined_temp] = 0

                        # Loop over controls j
                        for j in range(1, count_controls + 1):
                            col_in_brackets_j = f"in_brackets_{l}_{j}_XX"
                            if col_in_brackets_j not in df.columns:
                                df[col_in_brackets_j] = 0

                            # Loop over controls k
                            for k in range(1, count_controls + 1):
                                col_in_brackets_temp = f"in_brackets_{l}_{j}_{k}_temp_XX"
                                if col_in_brackets_temp in df.columns:
                                    df.drop(columns=col_in_brackets_temp, inplace=True)

                                coef_jk = dict_glob[f"inv_Denom_{l}_XX"][j-1, k-1]  # 1-based in Stata, 0-based here
                                mask = (df["d_sq_int_XX"] == l) & (df["F_g_XX"] >= 3)

                                df[col_in_brackets_temp] = np.where(
                                    mask,
                                    coef_jk * df[f"in_sum_{k}_{l}_XX"],
                                    0
                                )

                                # Sum over k
                                df[col_in_brackets_j] += df[col_in_brackets_temp]

                            # Subtract theta_d
                            coef_theta = np.array(dict_glob[f"coefs_sq_{l}_XX"]).reshape(-1, 1)[j-1, 0]  # Stata col 1 → pandas col 0
                            df[col_in_brackets_j] -= coef_theta

                            # Compute cross product with M^(+)
                            col_combined_j = f"combined{increase_XX}_temp_{l}_{j}_{i}_XX"
                            if col_combined_j in df.columns:
                                df.drop(columns=col_combined_j, inplace=True)

                            df[col_combined_j] = df[f"M{increase_XX}_{l}_{j}_{i}_XX"] * df[col_in_brackets_j]

                            # Sum over j
                            df[col_combined_temp] += df[col_combined_j]

                        # Final sum over status quo treatment
                        col_part2 = f"part2_switch{increase_XX}_{i}_XX"
                        if col_part2 not in df.columns:
                            df[col_part2] = 0
                        df[col_part2] += df[col_combined_temp]


            # 7. Sum U_Gg_var over time by group
            df[f"U_Gg{i}_var_XX"] = df.groupby("group_XX")[f"U_Gg{i}_temp_var_XX"].transform("sum")
            
            # 8. Adjustment for controls, if any
            if controls not in (None, "", []):  # check if controls are provided
                if increase_XX == 1:
                    df[f"U_Gg{i}_var_XX"] = (
                        df[f"U_Gg{i}_var_XX"] - df[f"part2_switch1_{i}_XX"]
                    )
                elif increase_XX == 0:
                    # CHANGE BELOW: minus sign included here
                    df[f"U_Gg{i}_var_XX"] = (
                        df[f"U_Gg{i}_var_XX"] - df[f"part2_switch0_{i}_XX"]
                    )


        # inside your loop over i:
        if normalized:
            # 1. construct sum_temp_XX
            mask = (
                (df["time_XX"] >= df["F_g_XX"]) &
                (df["time_XX"] <= df["F_g_XX"] - 1 + i) &
                (df["S_g_XX"] == increase_XX)
            )
            if continuous == 0:
                df["sum_temp_XX"] = np.where(
                    mask,
                    df["treatment_XX"] - df["d_sq_XX"],
                    np.nan
                )
            elif continuous >0:
                df["sum_temp_XX"] = np.where(
                    mask,
                    df["treatment_XX_orig"] - df["d_sq_XX_orig"],
                    np.nan
                )

            # 2. sum up by group
            df[f"sum_treat_until_{i}_XX"] = (
                df.groupby("group_XX")["sum_temp_XX"]
                .transform("sum")
            )

            # 3. drop the helper column
            df.drop(columns="sum_temp_XX", inplace=True)

            # 4. compute delta_D_i_cum_temp_XX
            N_val = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
            ratio = df["N_gt_XX"] / N_val
            switch_mask = df[f"distance_to_switch_{i}_XX"] == 1

            df[f"delta_D_{i}_cum_temp_XX"] = np.where(
                switch_mask,
                ratio * (
                    df["S_g_XX"] * df[f"sum_treat_until_{i}_XX"]
                    + (1 - df["S_g_XX"]) * (-df[f"sum_treat_until_{i}_XX"])
                ),
                np.nan
            )

            # 5. store the scalar in const
            dict_vars_gen[f"delta_norm_{i}_XX"] = df[f"delta_D_{i}_cum_temp_XX"].sum(skipna=True)



    Ntrendslin=1
    # 1. Compute Ntrendslin
    Ntrendslin = min(
        int(dict_vars_gen[f"N{increase_XX}_{i}_XX"])
        for i in range(1, int(l_u_a_XX + 1))
    )

    # 2. If linear trends requested and there's at least one valid path
    if trends_lin and Ntrendslin != 0:

        print(f"This is the l_u_a_XX {l_u_a_XX} value")
        print("accion extra")
        # Column names corresponding to `=l_u_a_XX'`
        col_TL       = f"U_Gg{int(l_u_a_XX)}_TL"
        col_var_TL   = f"U_Gg{int(l_u_a_XX)}_var_TL"
        col_XX       = f"U_Gg{int(l_u_a_XX)}_XX"
        col_var_XX   = f"U_Gg{int(l_u_a_XX)}_var_XX"

        # capture drop U_Gg`=l_u_a_XX'_TL
        # capture drop U_Gg`=l_u_a_XX'_var_TL
        for c in [col_TL, col_var_TL]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        # gen U_Gg`=l_u_a_XX'_TL = 0
        # gen U_Gg`=l_u_a_XX'_var_TL = 0
        df[col_TL]     = 0.0
        df[col_var_TL] = 0.0

        # forvalue i=1/`=l_u_a_XX' {
        #   replace U_Gg`=l_u_a_XX'_TL      = U_Gg`=l_u_a_XX'_TL      + U_Gg`i'_XX
        #   replace U_Gg`=l_u_a_XX'_var_TL  = U_Gg`=l_u_a_XX'_var_TL  + U_Gg`i'_var_XX
        # }
        # List all columns starting with 'U_Gg'
        cols_U_Gg = [col for col in df.columns if col.startswith("U_Gg")]
        print(cols_U_Gg)

        for i in range(1, int(l_u_a_XX + 1)):
            print(col_TL)
            df[col_TL]     = df[col_TL]     + df[f"U_Gg{i}_XX"]
            df[col_var_TL] = df[col_var_TL] + df[f"U_Gg{i}_var_XX"]

        # replace U_Gg`=l_u_a_XX'_XX      = U_Gg`=l_u_a_XX'_TL
        # replace U_Gg`=l_u_a_XX'_var_XX  = U_Gg`=l_u_a_XX'_var_TL
        df[col_XX]     = df[col_TL].copy()
        df[col_var_XX] = df[col_var_TL].copy()

    
    if placebo != 0:

        if l_placebo_u_a_XX >= 1:

            for i in range(1, int(l_placebo_u_a_XX + 1) ): 
                i=int(i)
                cols_to_drop = [
                    f"diff_y_pl_{i}_XX",
                    f"U_Gg_pl_{i}_temp_XX",
                    f"U_Gg_placebo_{i}_XX",
                    f"U_Gg_pl_{i}_temp_var_XX",
                    f"U_Gg_pl_{i}_var_XX",
                    f"mean_diff_y_pl_{i}_nd_sq_t_XX",
                    f"mean_diff_y_pl_{i}_d_sq_t_XX",
                    f"count_diff_y_pl_{i}_nd_sq_t_XX",
                    f"count_diff_y_pl_{i}_d_sq_t_XX",
                    f"dist_to_switch_pl_{i}_XX",
                    f"never_change_d_pl_{i}_XX",
                    f"N{increase_XX}_t_placebo_{i}_XX",
                    f"N{increase_XX}_t_placebo_{i}_g_XX",
                    f"N_gt_control_placebo_{i}_XX",
                    f"dummy_U_Gg_pl_{i}_XX",
                    f"never_change_d_pl_{i}_wXX",
                    f"dist_to_switch_pl_{i}_wXX",
                    f"dof_cohort_pl_{i}_ns_t_XX",
                    f"count_cohort_pl_{i}_ns_t_XX",
                    f"total_cohort_pl_{i}_ns_t_XX",
                    f"mean_cohort_pl_{i}_ns_t_XX",
                    f"dof_cohort_pl_{i}_s_t_XX",
                    f"dof_cohort_pl_{i}_s0_t_XX",
                    f"dof_cohort_pl_{i}_s1_t_XX",
                    f"dof_cohort_pl_{i}_s2_t_XX",
                    f"count_cohort_pl_{i}_s_t_XX",
                    f"count_cohort_pl_{i}_s0_t_XX",
                    f"count_cohort_pl_{i}_s1_t_XX",
                    f"count_cohort_pl_{i}_s2_t_XX",
                    f"total_cohort_pl_{i}_s_t_XX",
                    f"total_cohort_pl_{i}_s0_t_XX",
                    f"total_cohort_pl_{i}_s1_t_XX",
                    f"total_cohort_pl_{i}_s2_t_XX",
                    f"mean_cohort_pl_{i}_s_t_XX",
                    f"mean_cohort_pl_{i}_s0_t_XX",
                    f"mean_cohort_pl_{i}_s1_t_XX",
                    f"mean_cohort_pl_{i}_s2_t_XX",
                ]
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

                # 2) compute the long‐difference for placebo i
                df[f"diff_y_pl_{i}_XX"] = (
                    df.groupby("group_XX")["outcome_XX"].shift(2 * i)
                    - df.groupby("group_XX")["outcome_XX"].shift(i)
                )

                # 3) identify controls for placebo
                df[f"never_change_d_pl_{i}_XX"] = (
                    df[f"never_change_d_{i}_XX"] * df[f"diff_y_pl_{i}_XX"].notna()
                )
                check_cols = f"never_change_d_{i}_XX"
                idx = df[check_cols].isna()
                df.loc[ idx, f"never_change_d_pl_{i}_XX"] = np.nan


                # 4) weighted count of control groups
                df[f"never_change_d_pl_{i}_wXX"] = df[f"never_change_d_pl_{i}_XX"] * df["N_gt_XX"]
                if trends_nonparam is None:
                    cols_gr_sel = ["time_XX", "d_sq_XX"]
                else:
                    cols_gr_sel = ["time_XX", "d_sq_XX"] + trends_nonparam
                df[f"N_gt_control_placebo_{i}_XX"] = (
                    df
                    .groupby(cols_gr_sel)[f"never_change_d_pl_{i}_wXX"]
                    .transform("sum")
                )

                # # 5) distance‐to‐switch for placebo
                # col_dist = f"distance_to_switch_{i}_XX"
                # col_diff = f"diff_y_pl_{i}_XX"
                # col_res  = f"dist_to_switch_pl_{i}_XX"
                # df[col_res] = df[col_dist] * df[col_diff].notna().astype(int)
                # checkna = df[col_dist].isna()
                # df.loc[ checkna, col_res] = np.nan

                # 5) distance-to-switch for placebo
                col_dist = f"distance_to_switch_{i}_XX"
                col_diff = f"diff_y_pl_{i}_XX"
                col_N    = f"N_gt_control_placebo_{i}_XX"
                col_res  = f"dist_to_switch_pl_{i}_XX"

                # Indicator: diff_y_pl not missing AND N_gt_control_placebo > 0 AND not missing
                mask_valid = (
                    df[col_diff].notna() &
                    df[col_N].notna() &
                    (df[col_N] > 0)
                )

                # Multiply distance by the indicator (0/1)
                df[col_res] = df[col_dist] * mask_valid.astype(int)
                checkna = df[col_dist].isna() | df[col_diff].isna()
                df.loc[checkna, col_res] = np.nan


                # check_cols =[ f"distance_to_switch_{i}_XX",f"diff_y_pl_{i}_XX" ]
                # idx = df[check_cols].isna().sum(axis = 1)==2
                # df.loc[ idx, f"dist_to_switch_pl_{i}_XX"] = np.nan

                if same_switchers_pl:
                    df[f"dist_to_switch_pl_{i}_XX"] *= df["fillin_g_pl_XX"]
                df[f"dist_to_switch_pl_{i}_wXX"] = df[f"dist_to_switch_pl_{i}_XX"] * df["N_gt_XX"]

                # 6) counts by time
                df[f"N{increase_XX}_t_placebo_{i}_XX"] = (
                    df.groupby("time_XX")[f"dist_to_switch_pl_{i}_wXX"].transform("sum")
                )
                df[f"N{increase_XX}_t_placebo_{i}_dwXX"] = (
                    df.groupby("time_XX")[f"dist_to_switch_pl_{i}_XX"].transform("sum")
                )

                # 7) scalar N_{+|−,placebo,i}
                dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"] = 0
                dict_vars_gen[f"N{increase_XX}_dw_placebo_{i}_XX"] = 0
                n_placebo =  dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]
                n_dw_placebo =  dict_vars_gen[f"N{increase_XX}_dw_placebo_{i}_XX"]
                # ——— Loop over time and add up the period‐means ———
                for t in range(int(t_min_XX), int(T_max_XX + 1)):
                    # build your column names
                    col_p  = f"N{increase_XX}_t_placebo_{i}_XX"
                    col_dp = f"N{increase_XX}_t_placebo_{i}_dwXX"
                    
                    # mask for this period
                    mask = df['time_XX'] == t
                    
                    # add the mean of each column (skipna=True drops NaNs)
                    n_placebo    += df.loc[mask, col_p ].mean(skipna=True)
                    n_dw_placebo += df.loc[mask, col_dp].mean(skipna=True)

                # ——— (Optional) Put them in a dict if you want “dynamic” names ———
                dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"] = n_placebo
                dict_vars_gen[f"N{increase_XX}_dw_placebo_{i}_XX"] = n_dw_placebo
                

                # 8) group‐by‐cohort count
                if trends_nonparam is None:
                    cols_gr_sel = ["time_XX", "d_sq_XX"]
                else:
                    cols_gr_sel = ["time_XX", "d_sq_XX"] + trends_nonparam
                    
                df[f"N{increase_XX}_t_placebo_{i}_g_XX"] = (
                    df
                    .groupby(cols_gr_sel)[f"dist_to_switch_pl_{i}_wXX"]
                    .transform("sum")
                )

                # Only run if controls are provided
                if controls:
                    # 1) initialize intermediate adjustment column
                    df[f"part2_pl_switch{increase_XX}_{i}_XX"] = 0

                    # 2) loop over each control variable
                    for j, var in enumerate(controls, start=1):
                        j=int(j)
                        # a) long difference for placebo
                        df[f"diff_X_{j}_placebo_{i}_XX"] = (
                            df.groupby("group_XX")[var].shift(2 * i) - df.groupby("group_XX")[var].shift(i)
                        )
                        # b) scale by N_gt
                        df[f"diff_X{j}_pl_{i}_N_XX"] = df["N_gt_XX"] * df[f"diff_X_{j}_placebo_{i}_XX"]

                        # c) loop over each baseline-treatment level `l`
                        for l in levels_d_sq_XX:
                            l=int(l)
                            mask_l = df["d_sq_int_XX"] == l
                            denom = dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]
                            # compute m_{pl,g,l,j,i}
                            mcol = f"m{increase_XX}_pl_g_{l}_{j}_{i}_XX"
                            df[mcol] = (
                                ((df["T_g_XX"] - 2) >= i) & mask_l
                            ).astype(int) \
                            * (G_XX / denom) \
                            * (
                                df[f"dist_to_switch_pl_{i}_XX"]
                                - (df[f"N{increase_XX}_t_placebo_{i}_g_XX"] /
                                df[f"N_gt_control_placebo_{i}_XX"]) 
                                * df[f"never_change_d_pl_{i}_XX"]
                            ) \
                            * ((df["time_XX"] >= i+1) & (df["time_XX"] <= df["T_g_XX"])).astype(int) \
                            * df[f"diff_X{j}_pl_{i}_N_XX"]

                            # sum across t by group
                            summed = df.groupby("group_XX")[mcol].transform("sum")
                            df[f"m_pl{increase_XX}_{l}_{j}_{i}_XX"] = np.where(
                                df["first_obs_by_gp_XX"] == 1,
                                summed,
                                np.nan
                            )

                            # compute M_{pl,d,l,j,i}
                            df[f"M_pl{increase_XX}_{l}_{j}_{i}_XX"] = (
                                df[f"m_pl{increase_XX}_{l}_{j}_{i}_XX"].sum() / G_XX
                            )


                        for l in levels_d_sq_XX:
                            l = int(l)
                            # residualize diff_y_pl if needed
                            if dict_glob.get(f"useful_res_{l}_XX", 0) > 1:
                                coefs = np.array(dict_glob[f"coefs_sq_{l}_XX"]).reshape(-1, 1)  # numpy array, shape (n_controls,1)
                                df[f"diff_y_pl_{i}_XX"] = np.where(
                                    mask_l,
                                    df[f"diff_y_pl_{i}_XX"]
                                    - coefs[j-1, 0] * df[f"diff_X_{j}_placebo_{i}_XX"],
                                    df[f"diff_y_pl_{i}_XX"]
                                )
                                # initialize in‐brackets term for variance adjustment
                                df[f"in_brackets_pl_{l}_{j}_XX"] = 0


                # compute weighted placebo differences
                df[f"diff_y_pl_{i}_N_gt_XX"] = df[f"diff_y_pl_{i}_XX"] * df["N_gt_XX"]
                dof_ns_pl = f"dof_ns_pl_{i}_XX"
                df[dof_ns_pl] = (   ( df['N_gt_XX'] !=0 ) 
                                 & ( df[f"diff_y_pl_{i}_XX"].notnull() )
                                 & ( df[f'never_change_d_pl_{i}_XX'] == 1 )
                                 & ( df[f'N{increase_XX}_t_placebo_{i}_XX'] > 0 )
                                 & ( df[f'N{increase_XX}_t_placebo_{i}_XX'].notnull() ) ) * 1


                dof_s_pl = f"dof_s_pl_{i}_XX"
                df[ dof_s_pl ] = (
                    (df["N_gt_XX"] != 0) & df[f"dist_to_switch_pl_{i}_XX"] == 1
                ).astype(int)

                if trends_nonparam is None:
                    group_cols = ["d_sq_XX", "time_XX"]
                else:
                    group_cols = ["d_sq_XX", "time_XX"] + trends_nonparam

                mask_valid = df[dof_ns_pl]==1
                df[f"count_cohort_pl_{i}_ns_t_XX"] = (
                    df[mask_valid]
                    .groupby(group_cols)["N_gt_XX"]
                    .transform("sum")
                )
                df.loc[~mask_valid, f"count_cohort_pl_{i}_ns_t_XX"] = np.nan

                # numerator of the mean
                mask_num = df[dof_ns_pl]==1
                df[f"total_cohort_pl_{i}_ns_t_XX"] = (
                    df[mask_num]
                    .groupby(group_cols)[f"diff_y_pl_{i}_N_gt_XX"]
                    .transform("sum")
                )
                df.loc[~mask_num, f"total_cohort_pl_{i}_ns_t_XX"] = np.nan

                # mean
                df[f"mean_cohort_pl_{i}_ns_t_XX"] = (
                    df[f"total_cohort_pl_{i}_ns_t_XX"] /
                    df[f"count_cohort_pl_{i}_ns_t_XX"]
                )

                
                ## Counting number of groups for DOF adjustment
                col_dof_ns   = f"dof_ns_pl_{i}_XX"
                col_dof_coh_ns  = f"dof_cohort_pl_{i}_ns_t_XX"
                
                # Build group keys: d_sq_XX `trends_nonparam' time_XX
                if trends_nonparam is None:
                    group_keys = ["d_sq_XX", "time_XX"]
                else:
                    group_keys = ["d_sq_XX" ] + trends_nonparam + [ "time_XX"]
                if cluster is None or cluster == "":
                    df[col_dof_coh_ns] = np.where(
                        df[col_dof_ns] == 1,
                        df.groupby(group_keys, dropna=False)[col_dof_ns].transform("sum"),
                        np.nan
                    ).astype(float)

                # Case 2: with cluster -> number of unique clusters among rows with dof_ns==1
                else:
                    # temp series: cluster where dof_ns==1, else NaN
                    # cluster_dof_pl_`i'_ns_XX
                    idx = df[col_dof_ns] == 1
                    clust_dof = f'cluster_dof_pl_{i}_ns_XX'
                    df[ clust_dof ] = np.nan
                    df.loc[ idx, clust_dof ] = df.loc[idx, cluster]

                    mask = df[clust_dof].notna()
                    agg = (
                        df.loc[mask, group_keys + [clust_dof]]
                        .groupby(group_keys, as_index=False)[clust_dof].nunique()
                        .rename(columns={clust_dof: col_dof_coh_ns})
                    )
                    
                    df = df.merge(agg, on=group_keys, how="left")
                    mask = df[clust_dof].notna()
                    df.loc[~mask, col_dof_coh_ns] = np.nan

                # switchers cohort (C_k) demeaning
                mask_sw = df[f"dof_s_pl_{i}_XX"] == 1
                
                if trends_nonparam is None:
                    group_cols_sw = ["d_sq_XX", "F_g_XX", "d_fg_XX" ]
                else:
                    group_cols_sw = ["d_sq_XX", "F_g_XX", "d_fg_XX"] + trends_nonparam

                # denominator
                count_cohort_pl_stXX = f"count_cohort_pl_{i}_s_t_XX"
                df[ count_cohort_pl_stXX ] = np.nan
                df[ count_cohort_pl_stXX] = (
                    df.loc[mask_sw, :].groupby(group_cols_sw)["N_gt_XX"]
                    .transform("sum")
                )
                df.loc[~mask_sw, count_cohort_pl_stXX ] = np.nan

                # numerator
                df[f"total_cohort_pl_{i}_s_t_XX"] = (
                    df.loc[mask_sw, :].groupby(group_cols_sw)[f"diff_y_pl_{i}_N_gt_XX"]
                    .transform("sum")
                )
                df.loc[~mask_sw, f"total_cohort_pl_{i}_s_t_XX"] = np.nan

                # mean
                df[f"mean_cohort_pl_{i}_s_t_XX"] = (
                    df[f"total_cohort_pl_{i}_s_t_XX"] /
                    df[f"count_cohort_pl_{i}_s_t_XX"]
                )

                # Build group keys: d_sq_XX `trends_nonparam' time_XX
                if trends_nonparam is None:
                    group_keys = ["d_sq_XX", "F_g_XX", "d_fg_XX"]
                else:
                    group_keys = ["d_sq_XX", "F_g_XX", "d_fg_XX"] + trends_nonparam
                
                col_dof_cohs = f"dof_cohort_pl_{i}_s_t_XX"
                col_dof_ns = f"dof_s_pl_{i}_XX"
                if cluster is None or cluster == "":
                    mask = df[col_dof_ns] == 1
                    df[col_dof_cohs] =  df.loc[mask,:].groupby(group_keys, dropna=False)[col_dof_ns].transform("sum")
                    
                # Case 2: with cluster -> number of unique clusters among rows with dof_ns==1
                else:
                    # temp series: cluster where dof_ns==1, else NaN
                    idx = df[col_dof_ns] == 1
                    clust_dof = f'cluster_dof_pl_{i}_s_XX'
                    df[ clust_dof ] = np.nan
                    df.loc[ idx, clust_dof ] = df.loc[idx, cluster]

                    mask = df[clust_dof].notna()
                    agg = (
                        df.loc[mask, group_keys + [clust_dof]]
                        .groupby(group_keys, as_index=False)[clust_dof].nunique()
                        .rename(columns={clust_dof: col_dof_cohs})
                    )
                    
                    df = df.merge(agg, on=group_keys, how="left")
                    mask = df[clust_dof].notna()
                    df.loc[~mask, col_dof_cohs] = np.nan

                

                #  Modif Clément 19/6/2025:
			    #  If a switcher is the only one in their cohort or if a not-yet-switcher is the only one in their cohort, we demean wrt union of switchers and not-yet switchers.
                if trends_nonparam is None:
                    group_keys = ["d_sq_XX", "time_XX"]
                else:
                    group_keys = ["d_sq_XX", "time_XX"] + trends_nonparam

                print(group_keys)
                col_dof_s    = f"dof_s_pl_{i}_XX"
                col_dof_ns   = f"dof_ns_pl_{i}_XX"
                col_dof_any  = f"dof_ns_s_pl_{i}_XX"

                col_N        = "N_gt_XX"
                col_diffN    = f"diff_y_pl_{i}_N_gt_XX"

                col_count    = f"count_cohort_pl_{i}_ns_s_t_XX"
                col_total    = f"total_cohort_pl_{i}_ns_s_t_XX"
                col_mean     = f"mean_cohort_pl_{i}_ns_s_t_XX"
                # --- dof mask: (dof_s == 1) OR (dof_ns == 1) ---
                mask_dof = (df[col_dof_s] == 1) | (df[col_dof_ns] == 1)
                df[col_dof_any] = mask_dof.astype(float)  # numeric 1.0 / 0.0 like Stata
                maskna = df[col_dof_s].isna() | df[col_dof_ns].isna()
                df.loc[ maskna, col_dof_any] = np.nan

                # Group sums (aligned with the full df index)
                grp_sums = df.loc[mask_dof, :].groupby(group_keys, dropna=False).transform("sum")

                # Assign results and blank out rows not in the mask
                df[col_count] = grp_sums[col_N]
                df[col_total] = grp_sums[col_diffN]
                df.loc[~mask_dof, [col_count, col_total]] = np.nan
                df[col_mean] = df[col_total] / df[col_count]
                
                col_dof_coh = f"dof_cohort_pl_{i}_ns_s_t_XX"
                print(f'este es el {cluster}')
                if cluster is None or cluster == "":
                    mask = df[col_dof_any] == 1
                    df[col_dof_coh] =  df.loc[mask,:].groupby(group_keys, dropna=False)[col_dof_any].transform("sum")
                    
                # Case 2: with cluster -> number of unique clusters among rows with dof_ns==1
                else:
                    print(f'estoy corriendo el correcto code')
                    # temp series: cluster where dof_ns==1, else NaN
                    df = df.reset_index(drop =True)
                    col_dof_ns = f"dof_ns_s_pl_{i}_XX"
                    idx = df[col_dof_ns] == 1
                    clust_dof = f'cluster_dof_pl_{i}_ns_s_XX'
                    df[ clust_dof ] = np.nan
                    df.loc[ idx, clust_dof ] = df.loc[idx, cluster]
                    try:
                        df = df.drop(col_dof_coh, axis =1 )
                    except:
                        print(col_dof_coh)
                    mask = df[clust_dof].notna()
                    agg = df.loc[mask, group_keys + [clust_dof]].copy() \
                        .groupby(group_keys, as_index=False)[clust_dof].nunique() \
                        .rename(columns={clust_dof: col_dof_coh})
                    print(agg)
                    df = df.merge(agg, on=group_keys, how="left").reset_index(drop = True)
                    print(df.columns)
                    print(df[col_dof_coh].describe())
                    mask = df[clust_dof].notna()
                    df.loc[~mask, col_dof_coh] = np.nan



                df = compute_E_hat_gt_with_nans(df, i = i, type_sect = "placebo" )



                # Correcting when only we have one switcher
                df = compute_DOF_gt_with_nans(df, i = i, type_sect="placebo")
                


                df[f"dummy_U_Gg_pl_{i}_XX"] = (i <= df["T_g_XX"] - 1).astype(int)
                # inside your loop over i:
                N_placebo = dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]
                if N_placebo != 0:
                    # compute the temporary U_Gg_pl
                    df[f"U_Gg_pl_{i}_temp_XX"] = (
                        df[f"dummy_U_Gg_pl_{i}_XX"]
                        * (G_XX / N_placebo)
                        * df["N_gt_XX"]
                        * (
                            df[f"dist_to_switch_pl_{i}_XX"]
                            - (df[f"N{increase_XX}_t_placebo_{i}_g_XX"] /
                            df[f"N_gt_control_placebo_{i}_XX"])
                            * df[f"never_change_d_pl_{i}_XX"]
                        )
                        * df[f"diff_y_pl_{i}_XX"]
                        * ((df["time_XX"] >= i + 1) & (df["time_XX"] <= df["T_g_XX"]))
                    )

                    # sum by group
                    df[f"U_Gg_placebo_{i}_XX"] = (
                        df.groupby("group_XX")[f"U_Gg_pl_{i}_temp_XX"]
                        .transform("sum")
                        * df["first_obs_by_gp_XX"]
                    )

                    # count_pl_core
                    cond = (
                        (~df[f"U_Gg_pl_{i}_temp_XX"].isna() & (df[f"U_Gg_pl_{i}_temp_XX"] != 0))
                        | (
                            (df[f"U_Gg_pl_{i}_temp_XX"] == 0)
                            & (df[f"diff_y_pl_{i}_XX"] == 0)
                            & (
                                (df[f"dist_to_switch_pl_{i}_XX"] != 0)
                                | (
                                    (df[f"N{increase_XX}_t_placebo_{i}_g_XX"] != 0)
                                    & (df[f"never_change_d_pl_{i}_XX"] != 0)
                                )
                            )
                        )
                    )
                    df[f"count{i}_pl_core_XX"] = np.where(cond, df["N_gt_XX"], 0).astype(float)
                    sel_cols = [ f"U_Gg_pl_{i}_temp_XX", f"diff_y_pl_{i}_XX", f"dist_to_switch_pl_{i}_XX", f"never_change_d_pl_{i}_XX" ]
                    idx = df[ sel_cols ].isna().sum(axis = 1) == 4
                    # df.loc[ idx, f"count{i}_pl_core_XX"] = np.nan
                    # compute temp_var
                    df[f"U_Gg_pl_{i}_temp_var_XX"] = (
                        df[f"dummy_U_Gg_pl_{i}_XX"]
                        * (G_XX / N_placebo)
                        * (
                            df[f"dist_to_switch_pl_{i}_XX"]
                            - (df[f"N{increase_XX}_t_placebo_{i}_g_XX"] /
                            df[f"N_gt_control_placebo_{i}_XX"])
                            * df[f"never_change_d_pl_{i}_XX"]
                        )
                        * ((df["time_XX"] >= i + 1) & (df["time_XX"] <= df["T_g_XX"]))
                        * df["N_gt_XX"]
                        * df[f"DOF_gt_pl_{i}_XX"]
                        * (df[f"diff_y_pl_{i}_XX"] - df[f"E_hat_gt_pl_{i}_XX"])
                    )

                    # add control adjustments, if any
                    if controls is not None:
                        for l in levels_d_sq_XX:
                            l=int(l)
                            # initialize the combined term
                            df[f"combined_pl{increase_XX}_temp_{l}_{i}_XX"] = 0
                            for j in range(1, count_controls + 1):
                                j=int(j)
                                for k in range(1, count_controls + 1):
                                    k=int(k)
                                    inv_Denom = dict_glob[f'inv_Denom_{(l)}_XX']
                                    df[f"in_brackets_pl_{l}_{j}_XX"] += (
                                        inv_Denom[j-1][k-1]
                                        * df[f"in_sum_{k}_{l}_XX"]
                                        * ((df["d_sq_int_XX"] == l) & (df["F_g_XX"] >= 3))
                                    )
                                coefsq = np.array(dict_glob[f'coefs_sq_{(l)}_XX']).reshape(-1, 1)
                                df[f"in_brackets_pl_{l}_{j}_XX"] -= coefsq[j-1, 0]
                                df[f"combined_pl{increase_XX}_temp_{l}_{i}_XX"] += (
                                    df[f"M_pl{increase_XX}_{l}_{j}_{i}_XX"]
                                    * df[f"in_brackets_pl_{l}_{j}_XX"]
                                )

                            df[f"part2_pl_switch{increase_XX}_{i}_XX"] += np.where(
                                df["d_sq_int_XX"] == l,
                                df[f"combined_pl{increase_XX}_temp_{l}_{i}_XX"],
                                0
                            )

                    # ensure numeric
                    df[f"U_Gg_pl_{i}_temp_var_XX"] = df[f"U_Gg_pl_{i}_temp_var_XX"].astype(float)

                    # sum variance term by group
                    df[f"U_Gg_pl_{i}_var_XX"] = (
                        df.groupby("group_XX")[f"U_Gg_pl_{i}_temp_var_XX"]
                        .transform("sum")
                    )

                    if controls is not None:  # equivalent to if "`controls'" != ""
                        if increase_XX == 1:
                            df[f"U_Gg_pl_{i}_var_XX"] = (
                                df[f"U_Gg_pl_{i}_var_XX"] - df[f"part2_pl_switch1_{i}_XX"]
                            )

                        elif increase_XX == 0:
                            # Note: changed "+" to "-" as per the Stata comment
                            df[f"U_Gg_pl_{i}_var_XX"] = (
                                df[f"U_Gg_pl_{i}_var_XX"] - df[f"part2_pl_switch0_{i}_XX"]
                            )
                    
                if normalized:
                    # 1) compute a temporary treatment‐sum for placebo
                    cond = (
                        (df["time_XX"] >= df["F_g_XX"])
                        & (df["time_XX"] <= df["F_g_XX"] - 1 + i)
                        & (df["S_g_XX"] == increase_XX)
                    )
                    if continuous ==0:
                        df["sum_temp_pl_XX"] = np.where(
                            cond,
                            df["treatment_XX"] - df["d_sq_XX"],
                            np.nan
                        )
                    elif continuous > 0:
                        df["sum_temp_pl_XX"] = np.where(
                            cond,
                            df["treatment_XX_orig"] - df["d_sq_XX_orig"],
                            np.nan
                        )

                    # 2) aggregate by group
                    df[f"sum_treat_until_{i}_pl_XX"] = (
                        df.groupby("group_XX")["sum_temp_pl_XX"]
                        .transform("sum")
                    )
                    df.drop(columns="sum_temp_pl_XX", inplace=True)

                    # 3) compute the delta‐D cumulative term
                    N_placebo = dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]
                    df[f"delta_D_pl_{i}_cum_temp_XX"] = np.where(
                        df[f"dist_to_switch_pl_{i}_XX"] == 1,
                        (df["N_gt_XX"] / N_placebo)
                        * (
                            df["S_g_XX"] * df[f"sum_treat_until_{i}_pl_XX"]
                            + (1 - df["S_g_XX"]) * (-df[f"sum_treat_until_{i}_pl_XX"])
                        ),
                        np.nan
                    )

                    # 4) store the normalized placebo effect in your const dict
                    dict_vars_gen[f"delta_norm_pl_{i}_XX"] = df[f"delta_D_pl_{i}_cum_temp_XX"].sum(skipna=True)
        
        Ntrendslin_pl = 1
        if trends_lin:
            Ntrendslin_pl = min(
                int(dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"])
                for i in range(1, int(l_placebo_u_a_XX + 1))
            )
        print(f"{l_placebo_u_a_XX} number placebo")
        if trends_lin and Ntrendslin_pl != 0:
            print( f"Entramos a subsection placebo. This values {l_placebo_u_a_XX}")
            lp = int(l_placebo_u_a_XX)

            # Column names matching the Stata locals/macros
            col_TL        = f"U_Gg_pl_{lp}_TL"
            col_var_TL    = f"U_Gg_pl_{lp}_var_TL"
            col_placebo   = f"U_Gg_placebo_{lp}_XX"
            col_pl_var_XX = f"U_Gg_pl_{lp}_var_XX"

            # capture drop U_Gg_pl_`=l_placebo_u_a_XX'_TL
            # capture drop U_Gg_pl_`=l_placebo_u_a_XX'_var_TL
            for c in [col_TL, col_var_TL]:
                if c in df.columns:
                    df.drop(columns=c, inplace=True)

            # gen U_Gg_pl_`=l_placebo_u_a_XX'_TL = 0
            # gen U_Gg_pl_`=l_placebo_u_a_XX'_var_TL = 0
            df[col_TL]     = 0.0
            df[col_var_TL] = 0.0

            # forvalue i=1/`=l_placebo_u_a_XX' {
            #   replace U_Gg_pl_..._TL     = U_Gg_pl_..._TL     + U_Gg_placebo_`i'_XX
            #   replace U_Gg_pl_..._var_TL = U_Gg_pl_..._var_TL + U_Gg_pl_`i'_var_XX
            # }
            for i in range(1, int(lp + 1)):
                df[col_TL]     = df[col_TL]     + df[f"U_Gg_placebo_{i}_XX"].copy()
                df[col_var_TL] = df[col_var_TL] + df[f"U_Gg_pl_{i}_var_XX"].copy()

            # replace U_Gg_placebo_`=l_placebo_u_a_XX'_XX = U_Gg_pl_`=l_placebo_u_a_XX'_TL
            # replace U_Gg_pl_`=l_placebo_u_a_XX'_var_XX = U_Gg_pl_`=l_placebo_u_a_XX'_var_TL
            df[col_placebo]   = df[col_TL].copy()
            df[col_pl_var_XX] = df[col_var_TL].copy()


    
    if not trends_lin:
        # 1) Compute sum_N{increase_XX}_l_XX
        print(f"this increase {increase_XX}")
        total_key = f"sum_N{increase_XX}_l_XX"
        dict_vars_gen[total_key] = sum(
            dict_vars_gen[f"N{increase_XX}_{i}_XX"]
            for i in range(1, int(l_u_a_XX) + 1)
        )
        
        # 2) Initialize needed DataFrame columns
        for col in ["U_Gg_XX", "U_Gg_num_XX", "U_Gg_den_XX", "U_Gg_num_var_XX", "U_Gg_var_XX"]:
            df[col] = 0


        for i in range(1, int(l_u_a_XX) + 1):

            # Column names
            N_increase = dict_vars_gen[f"N{increase_XX}_{i}_XX"]  # e.g. Nincrease_XX_3_XX
            sum_N_increase = dict_vars_gen[f"sum_N{increase_XX}_l_XX"]  # denominator for weight
            delta_temp = f"delta_D_{i}_temp_XX"
            delta = f"delta_D_{i}_XX"
            delta_g = f"delta_D_g_{i}_XX"
            dist_to_switch = f"distance_to_switch_{i}_XX"

            # Only run if N_increase != 0
            if N_increase != 0:

                # 1. Compute weight
                dict_vars_gen[ f"w_{i}_XX" ] = N_increase / sum_N_increase

                # 2. Compute delta_D_temp
                if continuous == 0:
                    df[delta_temp] = np.where(
                        df[dist_to_switch] == 1,
                        df["N_gt_XX"] / N_increase *
                        ((df["treatment_XX"] - df["d_sq_XX"]) * df["S_g_XX"] +
                        (1 - df["S_g_XX"]) * (df["d_sq_XX"] - df["treatment_XX"])),
                        np.nan
                    )
                elif continuous > 0:
                    
                    # Build the denominator column name: N`=increase_XX'_`i'_XX  → e.g. "N5_2_XX"
                    den_col = f"N{increase_XX}_{i}_XX"

                    # Stata: if distance_to_switch_`i'_XX == 1
                    mask = df[f"distance_to_switch_{i}_XX"] == 1

                    # Initialize as missing (like Stata's .)
                    df[f"delta_D_{i}_temp_XX"] = np.nan

                    df[ f"delta_D_{i}_temp_XX"] = (
                        (df["N_gt_XX"] / df[den_col])
                        * (
                            (df["treatment_XX_orig"] - df["d_sq_XX_orig"]) * df[ "S_g_XX"]
                            + (1 - df["S_g_XX"]) * (df["d_sq_XX_orig"] - df[ "treatment_XX_orig"])
                        )
                    )
                    df.loc[~mask, f"delta_D_{i}_temp_XX" ] = np.nan


                # Replace missing with 0
                df[delta_temp] = df[delta_temp].fillna(0)

                # 3. Aggregate delta_D (sum over all obs)
                df[delta] = df[delta_temp].sum()

                # 4. Compute delta_D_g
                df[delta_g] = df[delta_temp] * (N_increase / df["N_gt_XX"])

                # 5. Drop temp
                df.drop(columns=[delta_temp], inplace=True)

                # 6. Update U_Gg_* numerators and denominators (group level)
                w_i = dict_vars_gen[ f"w_{i}_XX" ]
                
                df["U_Gg_num_XX"] = df.groupby("group_XX")["U_Gg_num_XX"].transform(
                    lambda x: x + w_i * df[f"U_Gg{i}_XX"]
                )
                df["U_Gg_num_var_XX"] = df.groupby("group_XX")["U_Gg_num_var_XX"].transform(
                    lambda x: x + w_i * df[f"U_Gg{i}_var_XX"]
                )
                df["U_Gg_den_XX"] = df.groupby("group_XX")["U_Gg_den_XX"].transform(
                    lambda x: x + w_i * df[delta]
                )

        # 4) Final U_Gg and its variance
        df["U_Gg_XX"] = df["U_Gg_num_XX"] / df["U_Gg_den_XX"]
        df["U_Gg_var_XX"] = df["U_Gg_num_var_XX"] / df["U_Gg_den_XX"]

        # 5) Propagate back any normalized deltas into const
        # update all existing entries in const to their current values
        # List of local variables generated in core that are needed later
    

    for e in list(const.keys()):
        if e in locals():
            const[e] = locals()[e]
        elif e in dict_vars_gen:
            const[e] = dict_vars_gen[e]
        else:
            const[e] = 0  # or leave it alone, depending on how you want to handle missing

    # update the rolling‐sum key
    sum_key = f"sum_N{increase_XX}_l_XX"
    if sum_key in locals():
        const[sum_key] = locals()[sum_key]
    elif sum_key in dict_vars_gen:
        const[sum_key] = dict_vars_gen[sum_key]

    # add normalized deltas
    if normalized:
        for i in range(1, int(l_u_a_XX + 1)):
            key = f"delta_norm_{i}_XX"
            if key in locals():
                const[key] = locals()[key]
            elif key in dict_vars_gen:
                const[key] = dict_vars_gen[key]

    # add placebo deltas
    if placebo != 0 and l_placebo_u_a_XX >= 1 and normalized:
        for i in range(1, int(l_placebo_u_a_XX + 1)):
            key = f"delta_norm_pl_{i}_XX"
            if key in locals():
                const[key] = locals()[key]
            elif key in dict_vars_gen:
                const[key] = dict_vars_gen[key]
    
    # I am changing const for dict_vars_gen
    # data = { 'df' : df, 'const': const }
    data = { 'df' : df, 'const': dict_vars_gen }
    return data