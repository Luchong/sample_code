import itertools
import warnings
from random import sample
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from shared_utils import breakdown_ts
from shared_utils import TreatmentCoder


class SyntheticControlUtils:
    def __init__(self):
        pass

    @staticmethod
    def region_split(
            region_list,
            treatment_group_ratio=0.5,
            target_ts=pd.DataFrame()):
        """
        split the region list to treatment and control, based on target time series correlation
        :param region_list:
        :param treatment_group_ratio:
        :param target_ts: the time series for target, used for split most correlated groups to T and C,
                        if leave blanked, then just random sample
        :return:treatment list and control list
        """

        # check if the inputs are right
        if treatment_group_ratio > 1 or treatment_group_ratio < 0:
            raise ValueError(
                'region split: treatment grouping ratio should between 0 and 1')
        if not region_list:
            raise ValueError(
                'region split: Empty region list for region split')

        # calculate the size of treatment and control region
        treatment_size = int(len(region_list) * treatment_group_ratio)
        control_size = len(region_list) - treatment_size
        region_list = list(set(region_list))  # remove duplicate

        treatment_list, control_list = [], []
        if target_ts.empty:
            # if there is no indicator, we just randomly split to treatment and
            # control
            treatment_list = sample(region_list, treatment_size)
            control_list = list(set(region_list) - set(treatment_list))
        else:
            # if there is indicator to calculate similarity we use correlation
            # to find the most similar item
            if set(target_ts.index) != set(region_list):
                raise Exception(
                    'region split: regional time series and region list do not match, please check')
            while len(treatment_list) < treatment_size and len(
                    control_list) < control_size:
                a = sample(region_list, 1)[0]
                # most correlated
                b = target_ts.T.corr()[a].sort_values().index[-2]
                treatment_list.append(a)
                control_list.append(b)
                region_list.remove(a)
                region_list.remove(b)
                target_ts.drop([a, b], axis=0, inplace=True)

            # through the rest of region into the right group
            if len(treatment_list) < treatment_size:
                treatment_list.extend(region_list)
            else:
                control_list.extend(region_list)
        return treatment_list, control_list

    @staticmethod
    def treatment_assign(
            region_list,
            treatment_options,
            valid_treatment_list,
            assignment_type='random'):
        """
        assign the treatment to group
        :param region_list: the region grouping list, output from region_split
        :param treatment_options: a list of possible treatment
        :param valid_treatment_list: a dict, with region as key and "not available for treatment" as values
        :param assignment_type: string, either be random or balanced
                            1.  if it is random, the logic would be random choose one from the validate treatment for
                                the region, the benefit is it is pure randomization, but the problem is the total
                                allocation of treatment would be extremely biased due to the bias in the validate
                                treatments
                            2.  if it is balanced, the logic would be that we force all the treatment to applied for
                                similar times (region received treatment i should be identical or the difference in
                                num of region should less than 1) this would not be pure randomization, but the
                                treatment given should be balanced.
        :return: dict, with region as key and treatment assigned as value
        """
        if len(region_list) < len(treatment_options):
            raise ValueError(
                'treatment assign: not enough regions to assign treatment')

        if assignment_type == 'random':
            assignment = {}
            for region in region_list:
                treatments = valid_treatment_list[region]
                assignment[region] = sample(treatments, 1)[0]
        elif assignment_type == 'balanced':
            # TODO: have not figure a better way of doing this
            # warning!!!: highly possible failing to find a solution
            # list all the possible combination of treatment
            treatment_candidates = treatment_options * \
                (len(region_list) // len(treatment_options))
            treatment_candidates = treatment_candidates + \
                sample(treatment_options, len(region_list) % len(treatment_options))
            shuffle(treatment_candidates)

            # treat regions with restriction differently with others
            restricted_region = list(pd.DataFrame(
                data=[[i, len(val)] for i, val in valid_treatment_list.items()
                      if i in region_list and len(val) < len(treatment_options)],
                columns=['region', 'validate_channel']).
                set_index('region').sort_values(by='validate_channel', ascending=True).index)

            rest_region = list(set(region_list) - set(restricted_region))

            assignment = {}
            # firstly deal with restricted regions
            for region in restricted_region:
                treatment_candidates = [
                    i for i in treatment_candidates if i in valid_treatment_list[region]]
                if not treatment_candidates:
                    raise ValueError(
                        'the restriction exclusion is too large to find solution.')
                assignment[region] = sample(treatment_candidates, 1)[0]
                treatment_candidates.remove(assignment[region])
            # and then randomly assign rest regions
            for region in rest_region:
                assignment[region] = sample(treatment_candidates, 1)[0]
                treatment_candidates.remove(assignment[region])
        else:
            raise ValueError(
                "treatment assign: not recognized assignment type, should be 'random' or 'balanced'")

        return assignment

    @staticmethod
    def generate_treatments(treatment, n=1, assignment_type='random'):
        """
        Generate a list of all possible treatments. The treatment group could potentially be
        one treatment or multiple treatments at the same time, this is specified in min_max_treatment_applied
        :param treatment: possible treatments
        :param n: the num of treatment one region could potentially receive.
                                            If it is [1,1], it means only one treatment is allowed for one region
                                            If it is [1,2], it means one region could receive one treatment or two
                                                treatments.
        :param assignment_type
        :return: a dict of treatment name and treatment channels
        example
        """
        # choose n treatment for combination
        if assignment_type == 'random':
            candidate = []
            for i in range(1, n + 1):
                candidate.extend(list(itertools.combinations(treatment, i)))
            treatment_group = [
                TreatmentCoder.name_encoder(
                    list(i)) for i in candidate]
        elif assignment_type == 'balanced':
            # TODO: have not figure out a better way to do this
            # I only need m = len(treatment) to fully filled result, so
            candidate = list(itertools.combinations(treatment, n))
            candidate_group = list(
                itertools.combinations(
                    candidate, len(treatment)))

            def is_balanced(s1):
                check = list(itertools.chain(*s1))
                return len({check.count(x) for x in set(check)})

            # screen for the balanced one
            candidate_group = [i for i in candidate_group if is_balanced(i)]
            # choose one
            candidate_group = sample(candidate_group, 1)[0]
            treatment_group = [
                TreatmentCoder.name_encoder(
                    list(i)) for i in candidate_group]
        else:
            raise ValueError(
                "generate treatment: not recognized assignment type, should be 'random' or 'balanced'")
        return treatment_group

    @staticmethod
    def cal_treatment_effects(treatment_region_ts,
                              control_region_ts,
                              experiment_start,
                              validation_start,
                              automatic_choose_control,
                              automatic_choose_level_parameter,
                              include_confidence_interval,
                              alpha=0.05,
                              model_name='BSTS',
                              **kwargs):
        """
        calculate the treatment effects
        :param model_name: string, BSTS or Abadie
        :param treatment_region_ts: dataframe, the ts for all treatment regions, t*m
        :param control_region_ts: dataframe, the ts for all control regions,  t*n
        :param experiment_start: the date experiment starts
        :param validation_start: the date validation period starts
        :param automatic_choose_control: whether choose control region using validation period
        :param automatic_choose_level_parameter: whether to choose level parameter using validation period
        :param include_confidence_interval: whether to include confidence interval in reports
        :param alpha: the threshold for significant, default 0.05 = 0.95 sig
        :param kwargs: parameters for unobserved_component
        :return: two dataframe, with the cumulative impact and ci by region and by channel
        """
        # breakdown to training, validation and prediction data
        pre_t, post_t = breakdown_ts(treatment_region_ts, experiment_start)
        pre_c, post_c = breakdown_ts(control_region_ts, experiment_start)

        result_by_region = dict()
        result_summary, resids = pd.DataFrame(), pd.DataFrame(index=pre_t.index)

        if model_name == 'BSTS':
            for region in treatment_region_ts.columns:
                module = BSTSUtils()
                # build model and fit
                module.build(
                    y=pre_t[[region]],
                    x=pre_c,
                    return_model=False,
                    automatic_choose_control=automatic_choose_control,
                    validation_starts=validation_start,
                    automatic_choose_level_params=automatic_choose_level_parameter,
                    **kwargs)

                resids = resids.assign(**{region: module.fitted_models.resid})

                # prediction
                _ = module.predict(
                    x=post_c,
                    include_confidence_interval=include_confidence_interval,
                    alpha=alpha)

                # calculate cumulative impact
                tmp_region, one_line_summary = module.get_treatment_effects(
                    include_confidence_interval=include_confidence_interval,
                    y_post=post_t[[region]],
                    y_pre=pre_t[[region]],
                    x_post=post_c)

                # write to dict
                result_by_region[region] = tmp_region
                if result_summary.empty:
                    result_summary = pd.DataFrame({region: one_line_summary})
                else:
                    result_summary[region] = pd.DataFrame({region: one_line_summary})

                # TODO think about how to combine treatments
        else:
            raise NotImplementedError

        return result_by_region, result_summary.T, resids


class BSTSUtils:
    """
    original Bayes time series prediction .

    """
    ACTUAL_COL_NAME = 'actual'
    PRED_COL_NAME = 'prediction'
    TE_COL_NAME = 'daily_treatment_effects'
    CUMULATIVE_TE_COL_NAME = 'cumulative_treatment_effects'
    MAX_ITER = 1000
    N_SIMS = 500
    MIN_CONTROL = 3
    LAST_DATE_OFFSET = -1  # use this if we want to automatically exclude last several days. for now we use the last
    # day which is "yesterday"

    def __init__(self):
        self.fitted_models = None
        self.model_args = {}
        self.result = pd.DataFrame()
        self.chosen_region = []
        self.params = pd.DataFrame()
        self.average_treatment_effects = dict()

    @staticmethod
    def _check_inputs(y, x):
        """
        TODO: add more checks
        Check the input is validate
        :return: None
        """
        if (not isinstance(y, pd.DataFrame)) or (
                not isinstance(x, pd.DataFrame)):
            raise ValueError('Wrong y or x, please use data frame')
        if set(y.index) != set(x.index):
            raise ValueError('please match x and y')

    @staticmethod
    def _check_params(model_args):
        """
        automatic check params
        :param model_args: input params
        :return: None
        """
        # possible params
        p_param = ['level', 'trend', 'seasonal',
                   'freq_seasonal', 'cycle', 'autoregressive',
                   'irregular', 'stochastic_level', 'stochastic_trend',
                   'stochastic_seasonal', 'stochastic_freq_seasonal',
                   'stochastic_cycle', 'damped_cycle', 'cycle_period_bounds']
        for k, v in model_args.items():
            if k not in p_param:
                raise ValueError(f'not recognized params {v}')
            if k == 'seasonal' and v != 7:
                warnings.warn(f'not used weekly as seasonal, use {v} days')

    @staticmethod
    def get_model(y, x, **kwargs):
        """
        return the current model
        :param y: array
        :param x: array
        :param kwargs: model parameter
        :return: UnonbservedComponents model object
        """
        return UnobservedComponents(y,
                                    exog=x,
                                    **kwargs)

    def build(self, y, x,
              return_model=False,
              automatic_choose_control=True,
              validation_starts=None,
              automatic_choose_level_params=True,
              verbose=False,
              **kwargs):
        """
        Build the bsts model
        :param y: dataframe, nX1 the pre-experiment time series to treatment region
        :param x: dataframe, nXm the block pre-experiment time series of control regions
        :param return_model: Boolean, whether to return the model result
        :param automatic_choose_control: bool whether to choose control regions using stepwise
        :param validation_starts: string, the start date for validation period. need to be '2019-03-01' format
        :param automatic_choose_level_params: bool whether to choose level parameters
        :param verbose: whether to show automatic choose progress
        :param kwargs: other parameters for unobserved model. have 4 defaults set up
                level, cycle, damped_cycle and stochastic cycle.
                If need to visualize, check the following page
              http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
                If need to check the full list of params, check the following page
            https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
        :return: model or None
        """
        self._check_inputs(y, x)

        # set up model parameter and regions
        default_args = {'seasonal': 7}   # force a weekly trend
        if not automatic_choose_level_params:
            default_args.update({'level': 'strend'})

        # default choose all regions
        self.chosen_region = x.columns

        # read in params
        if len(default_args) > 0:
            for k, v in default_args.items():
                self.model_args[k] = v
        # overwrite and add new args
        for k, v in kwargs.items():
            self.model_args[k] = self.model_args.get(k, v)

        # check params

        # choose level params automatically
        if 'level' in self.model_args.keys():
            # remove current
            required = self.model_args.pop('level', None)
            warnings.warn(f'automatic to determine level param, so not use input {required}')
        if automatic_choose_level_params:
            y_train, y_validate = breakdown_ts(y, validation_starts)
            x_train, x_validate = breakdown_ts(x, validation_starts)
            level_params = self.choose_level_parameter(y_train, x_train, y_validate, x_validate, verbose,
                                                       **self.model_args)
            self.model_args['level'] = level_params
        # choose control regions
        if automatic_choose_control:
            y_train, y_validate = breakdown_ts(y, validation_starts)
            x_train, x_validate = breakdown_ts(x, validation_starts)
            self.chosen_region = self.step_wise(y_train, x_train, y_validate, x_validate, verbose,
                                                **self.model_args)

        # fit
        # TODO: double check if this will give us same result as R package
        self.fitted_models = self.get_model(y.values,
                                            x[self.chosen_region].values,
                                            **self.model_args).fit(maxiter=self.MAX_ITER)
        if return_model:
            return self.fitted_models

    def predict(self, x, include_confidence_interval=True, alpha=0.05):
        """
        Predict the counter factual afterward
        :param x: n2Xm the block post-experiment time series of control regions
        :param include_confidence_interval: Bool, whether to include confidence interval in result
        :param alpha: (float, optional) – The significance level for the
        confidence interval. ie., The default alpha = .05 returns a 95% confidence interval.
        :return: dataframe
        """
        predictions = self.fitted_models.get_forecast(steps=x.shape[0],
                                                      exog=x[self.chosen_region].values)
        self.result = pd.DataFrame(data=predictions.predicted_mean,
                                   index=x.index,
                                   columns=[self.PRED_COL_NAME])

        if include_confidence_interval:
            # add confidence interval
            ci = predictions.conf_int(alpha)
            if isinstance(ci, np.ndarray):  # <- problem in returning different in different version of package
                lower_ci = ci[:, 0]
                upper_ci = ci[:, 1]
            else:
                lower_ci = ci.loc[:, 'lower y'].values
                upper_ci = ci.loc[:, 'upper y'].values
            structs = {
                self.PRED_COL_NAME + '_lower_ci': lower_ci,
                self.PRED_COL_NAME + '_upper_ci': upper_ci}
            self.result = self.result.assign(**structs)
        return self.result

    def get_treatment_effects(
            self,
            y_post,
            x_post,
            y_pre,
            include_confidence_interval=True,
            alpha=0.05):
        """
        Calculate the treatment effects
        :param y_post: dataframe, x in experiment
        :param x_post: dataframe, y in experiment
        :param y_pre:  dataframe, y pre. to add to the result
        :param include_confidence_interval:  whether to include confidence interval
        :param alpha:  the critical values. 0.05 => 95% confidence level
        :return: two obj, the first one is daily effects, the 2nd one is dict w. summary stats
        """

        if self.result.empty:
            raise ValueError('Please ran the model first')

        # treatment effects
        te = y_post.iloc[:, 0] - self.result[self.PRED_COL_NAME]
        te_cum = te.cumsum()
        structs = {self.ACTUAL_COL_NAME: y_post,
                   self.TE_COL_NAME: te,
                   self.CUMULATIVE_TE_COL_NAME: te_cum}

        if include_confidence_interval:

            structs.update({
                self.TE_COL_NAME + '_lower_ci': y_post.iloc[:, 0] - self.result[self.PRED_COL_NAME + '_upper_ci'],
                self.TE_COL_NAME + '_upper_ci': y_post.iloc[:, 0] - self.result[self.PRED_COL_NAME + '_lower_ci']})

            # Confidence interval of the cumulative sum
            # simulate n time series
            simulated_y = self.simulated_ts(y_post=y_post, x_post=x_post[self.chosen_region], n_sims=self.N_SIMS)
            # calculate the simulated cumulative treatment effects
            cum_te = (- simulated_y + np.repeat(y_post.values, self.N_SIMS, axis=1)).cumsum(axis=0)
            # calculate percentile
            post_cum_pred_lower, post_cum_pred_upper = np.percentile(
                cum_te,
                [(alpha / 2) * 100, (1 - alpha / 2) * 100],
                axis=1
            )

            structs.update({self.CUMULATIVE_TE_COL_NAME + '_lower_ci': post_cum_pred_lower,
                            self.CUMULATIVE_TE_COL_NAME + '_upper_ci': post_cum_pred_upper})
        self.result = self.result.assign(**structs)
        self.result = y_pre.rename(columns={y_pre.columns[0]: self.ACTUAL_COL_NAME}).append(self.result)

        avg_te = (- simulated_y + np.repeat(y_post.values, self.N_SIMS, axis=1)).mean(axis=0)
        post_avg_pred_lower, post_avg_pred_upper = np.percentile(avg_te, [(alpha / 2) * 100, (1 - alpha / 2) * 100])

        self.average_treatment_effects = {self.ACTUAL_COL_NAME: y_post.values.mean(),
                                          self.PRED_COL_NAME: self.result[self.PRED_COL_NAME].mean(),
                                          self.TE_COL_NAME: te.mean(),
                                          self.TE_COL_NAME + '_st': avg_te.std(),
                                          self.TE_COL_NAME + '_upper_ci': post_avg_pred_upper,
                                          self.TE_COL_NAME + '_lower_ci': post_avg_pred_lower,
                                          self.CUMULATIVE_TE_COL_NAME: te_cum.iloc[self.LAST_DATE_OFFSET],
                                          self.CUMULATIVE_TE_COL_NAME + '_upper_ci':
                                              post_cum_pred_upper[self.LAST_DATE_OFFSET],
                                          self.CUMULATIVE_TE_COL_NAME + '_lower_ci':
                                              post_cum_pred_lower[self.LAST_DATE_OFFSET],

                                          }

        return self.result, self.average_treatment_effects

    def step_wise(self, y, x, y_validate, x_validate, verbose=False, **kwargs):
        """
        use backward step wise to choose best control regions
        :param y: fitting y
        :param x: fitting x
        :param y_validate:  validation y
        :param x_validate: validation x
        :param verbose: whether to show calculation process
        :return: list, string of region name
        """
        # generate a set of possible result
        best_solution, reach_to_min_control = False, False
        cols = x.columns
        while (not best_solution) and (not reach_to_min_control):
            cols = x.columns
            score = dict()
            score[0] = self.get_validation_score(y, x, y_validate, x_validate, **kwargs)
            for i, region in enumerate(cols):
                # calculate score for leaving out one region
                score[i + 1] = self.get_validation_score(y,
                                                         x.drop(columns=region),
                                                         y_validate,
                                                         x_validate.drop(columns=region),
                                                         **kwargs)
            models = {j: i for i, j in score.items()}
            min_score = min(score.values())
            min_model = models[min_score]
            if min_model == 0:
                best_solution = True
            elif len(cols) - 1 == self.MIN_CONTROL:
                reach_to_min_control = True
                cols = list(cols)
                r_remove = cols.pop(min_model - 1)
                if verbose:
                    print(f'remove:{r_remove}')
            else:
                x = x.drop(columns=cols[min_model - 1])
                x_validate = x_validate.drop(columns=cols[min_model - 1])
                if verbose:
                    print(f'remove:{cols[min_model - 1]}')
        if verbose:
            print('final control list is {}'.format(','.join(cols)))
        return cols

    def choose_level_parameter(self, y, x, y_validate, x_validate, verbose=False, **kwargs):
        """
        function to automatically select level parameters
        :param y: dataframe training y
        :param x: dataframe training x
        :param y_validate: dataframe validation y
        :param x_validate: dataframe validation x
        :param verbose: whether to show calculation process
        :param kwargs: other parameters for unobservedcomponents
        :return: parameter for level
        """
        possible_level_params = ['dconstant',  # Deterministic constant
                                 'llevel',  # Local level
                                 'rwalk',  # Random walk
                                 'dtrend',  # Deterministic trend
                                 'lldtrend',  # Local linear deterministic trend
                                 'rwdrift',   # Random walk with drift
                                 'lltrend',   # Local linear trend
                                 'strend',   # Smooth trend
                                 'rtrend']   # Random trend
        score = dict()
        for i, level_param in enumerate(possible_level_params):
            # calculate score for leaving out one region
            score[level_param] = self.get_validation_score(y,
                                                           x,
                                                           y_validate,
                                                           x_validate,
                                                           level=level_param,
                                                           **kwargs)
        models = {j: i for i, j in score.items()}
        if verbose:
            print(score)
        min_score = min(score.values())
        min_model = models[min_score]
        if verbose:
            print(f'level parameter to use {min_model}')
        return min_model

    def get_validation_score(self, y, x, y_validate, x_validate, **kwargs):
        """
        calculate the score for validation
        :param y: fitting y
        :param x: fitting x
        :param y_validate:  validation y
        :param x_validate: validation x
        :return: mean squared error
        """
        tmp_valid = self.get_model(y.values, x.values, **kwargs)\
            .fit(maxiter=self.MAX_ITER)\
            .forecast(steps=x_validate.shape[0], exog=x_validate.values)

        return np.sqrt(((y_validate.values.T - tmp_valid) ** 2).sum())

    def simulated_ts(self, y_post, x_post, n_sims):
        """
        Use the internal simulator to run the n simulations
        there is some small technical tricks to run this, for detail, please check this post
        https://stackoverflow.com/questions/51881148/simulating-time-series-with-unobserved-components-model
        :param y_post: dataframe, post y
        :param x_post: dataframe, post x
        :param n_sims: number of simulation
        :return: dataframe of n simulations
        """
        simulations = pd.DataFrame(index=y_post.index)
        # set up a blank model
        mod1 = self.get_model(np.zeros(y_post.shape), x_post[self.chosen_region].values, **self.model_args)
        # set up initiate states
        p_state = self.fitted_models.predicted_state[..., -1]
        p_state_cov = self.fitted_models.predicted_state_cov[..., -1]
        # get params
        params = self.fitted_models.params
        for c in range(n_sims):
            # set up a random initial state
            initial_state = np.random.multivariate_normal(p_state, p_state_cov)
            # run simulator
            sim = mod1.simulate(params, y_post.shape[0], initial_state=initial_state)
            # get simulation result
            simulations = simulations.assign(**{f'sim_{c}': sim})
        return simulations

    def get_in_sample_pred(self):
        """
        get in sample prediction
        :return: Dataframe
        """
        return self.fitted_models.get_prediction()

    @classmethod
    def plot(cls, experiment_starts_index, result):
        """
        Plot the result
        :param experiment_starts_index: int, the numeric index of experiment starts
        :param result: the output results from get_cumulative_impact_m
        :return: chart
        """
        # combine ts
        plt.figure(figsize=(20, 15))
        # chart1
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(result[cls.ACTUAL_COL_NAME], label=cls.ACTUAL_COL_NAME)
        plt.plot(result[cls.PRED_COL_NAME].iloc[experiment_starts_index:], 'r--', linewidth=2, label='counter-factual')
        plt.axvline(experiment_starts_index, c='k', linestyle='--')
        plt.fill_between(
            result.index[experiment_starts_index:],
            result[cls.PRED_COL_NAME + '_lower_ci'].iloc[experiment_starts_index:],
            result[cls.PRED_COL_NAME + '_upper_ci'].iloc[experiment_starts_index:],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(loc='upper left')
        plt.title('Observation vs counter-factual prediction')

        # chart2
        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(result.index, result[cls.TE_COL_NAME], 'r--', linewidth=2)
        plt.plot(result.index, np.zeros(result.shape[0]), 'g-', linewidth=2)
        plt.axvline(experiment_starts_index, c='k', linestyle='--')
        plt.fill_between(
            result.index,
            result[cls.TE_COL_NAME + '_lower_ci'],
            result[cls.TE_COL_NAME + '_upper_ci'],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([result.index[0], result.index[-1], None, None])
        ax2.set_xticklabels(result.index, rotation=45)
        plt.locator_params(axis='x', nbins=min(1, result.shape[0]))
        plt.title('Point-wise Impact')

        # chart3
        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(result.index, result[cls.CUMULATIVE_TE_COL_NAME], 'r--', linewidth=2)
        plt.plot(result.index, np.zeros(result.shape[0]), 'g-', linewidth=2)
        plt.axvline(experiment_starts_index, c='k', linestyle='--')
        plt.fill_between(
            result.index,
            result[cls.CUMULATIVE_TE_COL_NAME + '_lower_ci'],
            result[cls.CUMULATIVE_TE_COL_NAME + '_upper_ci'],
            facecolor='gray', interpolate=True, alpha=0.25,
        )
        plt.axis([result.index[0], result.index[-1], None, None])
        ax3.set_xticklabels(result.index, rotation=45)
        plt.locator_params(axis='x', nbins=min(1, result.shape[0]))
        plt.title('Cumulative Impact')
        plt.xlabel('$T$')
        plt.show()
        return ax1, ax2, ax3


