"""
"""

import collections
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_py.main.scripts.utils import Utils


class Visualizer(Utils):
    """
    A base class to create charts.

    Args:

    """
    def __init__(self):
        self.base_path = None
        self.outputs   = collections.defaultdict(list)

        self.figs = {}

    def output_results(self, save_data):
        """
        The main visualizer function

        Purpose is to output results from the object
        """
        return self._output_results(save_data)

    def make_time_series_fig(self, df,
                                   cols,
                                   xlabel=[],
                                   ylabel=[],
                                   legend=False,
                                   xlim='all',
                                   ylim=[],
                                   path=[]):
        """
        makes a time series figure from a dataframe and specified columns
        """

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        for col in cols:

            data = df.loc[:, col].astype(float)
            data.plot(kind='line', ax=ax, label=col)

            if ylim:
                ax.set_ylim(ylim)

            if xlabel:
                plt.xlabel(xlabel)

            if ylabel:
                plt.ylabel(ylabel)

            if legend:
                plt.legend()

            if xlim == 'last_week':
                start = df.index[-7 * 24 * 12]
                end = df.index[-1]

            if xlim == 'last_month':
                start = df.index[-30 * 24 * 12]
                end = df.index[-1]

            if xlim == 'all':
                start = df.index[0]
                end = df.index[-1]

            ax.set_xlim([start, end])

        if path:
            self.ensure_dir(path)
            fig.savefig(path)

        return fig

    def make_panel_fig(self, df,
                             panels,
                             xlabels,
                             ylabels,
                             shape,
                             xlim='all',
                             ylims=[],
                             path=None):
        """
        makes a panel of time series figures
        """

        assert len(panels) == len(xlabels)
        assert len(panels) == len(ylabels)
        assert shape[0] * shape[1] == len(panels)

        fig, axes = plt.subplots(nrows=shape[0],
                                 ncols=shape[1],
                                 figsize=(20, 20),
                                 sharex=True)

        for i, (ax, panel) in enumerate(zip(axes.flatten(),
                                            panels)):

            for col in panel:
                data = df.loc[:, col].astype(float)
                data.plot(kind='line', ax=ax, label=col)

                if ylims:
                    ax.set_ylim(ylims[i])

                ax.set_xlabel(xlabels[i])
                ax.set_ylabel(ylabels[i])
                ax.legend()

                if xlim == 'last_week':
                    start = df.index[-7 * 24 * 12]
                    end = df.index[-1]

                if xlim == 'last_month':
                    start = df.index[-30 * 24 * 12]
                    end = df.index[-1]

                if xlim == 'all':
                    start = df.index[0]
                    end = df.index[-1]

                ax.set_xlim([start, end])

        if path:
            self.ensure_dir(path)
            fig.savefig(path)

        return fig


class Eternity_Visualizer(Visualizer):
    """
    A class to join together data generated by the agent and environment
    """
    def __init__(self, episode,
                       agent,
                       env,
                       results_path='results/'):

        super().__init__()

        self.env = env
        self.agent = agent
        self.episode = episode

        self.base_path_agent = os.path.join(results_path)
        self.base_path_episodes = os.path.join(results_path, 'episodes')

        #  pull out the data
        print('Eternity visualizer is pulling data out of the agent')
        self.agent_memory = self.agent.memory.output_results()
        print('Eternity visualizer is pulling data out of the environment')
        self.env_info = self.env.output_results()

        self.state_ts = self.env.state_ts
        self.observation_ts = self.env.observation_ts

        #  use the index from the state_ts for the other len(total_steps) dataframes
        idx = pd.to_datetime(self.state_ts.index)
        dfs = [self.env_info['dataframe']]

        for df in dfs:
            df.index = idx

    def write_data_to_disk(self):
        print('saving env dataframe')
        save_df(self.env_info['dataframe'],
                os.path.join(self.base_path_episodes, 'env_history_{}.csv'.format(self.episode)))

        print('saving state dataframe')
        save_df(self.state_ts,
                os.path.join(self.base_path_episodes, 'state_ts_{}.csv'.format(self.episode)))

        print('saving memory steps dataframe')
        save_df(self.agent_memory['dataframe_steps'],
                os.path.join(self.base_path_agent, 'agent_df_steps.csv'))

        print('saving memory episodic dataframe')
        save_df(self.agent_memory['dataframe_episodic'],
                os.path.join(self.base_path_agent, 'agent_df_episodic.csv'))
        return None


    def _output_results(self, save_data):
        """
        Generates results
        """
        def save_df(df, path):
            self.ensure_dir(path)
            df.to_csv(path)

        print('saving the figures')

        #  iterate over the figure dictionary from the env visualizer
        #  this exists to allow env specific figures to be collected by
        #  the Eternity_Visualizer

        #  TODO similar thing for agent!
        # for make_fig_fctn, fig_name in self.env_figures.items():
        #     self.figures[fig_name] = make_fig_fctn(self.env_info,
        #                                            self.base_path_episodes)

        self.figs['panel'] = self.make_panel_fig(df=self.agent_memory['dataframe_episodic'],
                                                    panels=[['reward', 'cum_max_reward'],
                                                            ['rolling_mean']],
                                                    xlabels=['Episode',
                                                             'Episode'],
                                                    ylabels=['Total reward per episode',
                                                             'Rolling average reward per episode'],
                                                    shape=(2, 1),
                                                    path=os.path.join(self.base_path_agent, 'panel.png'))

        self.figs['last_ep'] = self.make_panel_fig(df=self.env_info['dataframe'],
                                                    panels=[['gross_rate'],
                                                            ['new_charge'],
                                                            ['electricity_price']],
                                                    xlabels=['Episode',
                                                             'Episode',
                                                             'Episode'],
                                                    ylabels=['Gross rate of charge/discharge [MW]',
                                                             'Battery charge level at end of step [MWh]',
                                                             'Electricity price [$/MWh]'],
                                                    shape=(3, 1),
                                                    path=os.path.join(self.base_path_agent, 'last_ep.png'))

        for var, df in self.agent_memory['agent_stats'].items():

            if var == 'training Q targets':
                hist, ax = plt.subplots(1, 1, figsize=(20, 20))
                df.loc[:,var].plot(kind='hist', bins=10, ax=ax)
                hist.savefig(os.path.join(self.base_path_agent,var+'.png'))

            else:
                self.figs[var] = self.make_time_series_fig(df=df,
                                                           cols=[var],
                                                           path=os.path.join(self.base_path_agent,var+'.png'))


        if save_data:
            self.write_data_to_disk()
