"""
The callbacks submodule defines callbacks to be used during the training
of the network.
"""

from datetime import datetime
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from keras import backend as K
from tensorflow.keras.callbacks import Callback


class TimeInfoCallback(Callback):
    """Reports time information during training, test, and predict of the network.
    This callback allows to report some time information related to 
    the network during the three phases: training, test, prediction.
    The information is printed out in the terminal.
    Usage with `fit()` API:
    ```python
        model.fit(
        X, y,
        callbacks=TimeInfoCallback())
    """

    def on_train_begin(self, logs=None):
        """Called when train of the network begins.
        This function prints out time informtion related to the
        beginning of the training.
        Args:
            - logs: dictionary containing metrics and loss
        """
        self.train_start_time = datetime.now()
        time_now = self.train_start_time.strftime('%H:%M:%S %d/%m/%Y')
        print(f'Started training at {time_now}\n')

    def on_train_end(self, logs=None):
        """Called when train of the network ends.
        This function prints out time informtion related to the
        end of the training.
        Args:
            - logs: dictionary containing metrics and loss
        """
        time_now = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
        print(f'Training completed at {time_now}')

    def on_epoch_end(self, epoch, logs=None):
        """Called when an epoch ends to print out time information.
        This function prints out the total time required for the training up to this
        epoch.
        Args:
            - logs: dictionary containing metrics and loss
        """
        seconds_training = (datetime.now() - self.train_start_time).total_seconds()
        hours, remainder = divmod(seconds_training, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f' {math.floor(hours):02d}:{math.floor(minutes):02d}:{math.floor(seconds):02d} spent for training')


class MetricsPlot(Callback):
    """Callback to plot metrics obtained during the training of the network.
    This call
    Args:
        output_dir: output directory where the plot file will be saved
        metrics: list of metrics to be plot. The names of the metrics must be
            contained in the `logs` dictionary
        file_format: format for saving the plot. Default to 'png'
    Usage with `fit()` API:
    ```python
    model.fit(
        X,y,
        callbacks=MetricsPlot(output_dir='/path/to/folder', metrics='loss'))
    """

    def __init__(self, output_dir=None, metrics=None, file_format='png'):
        super(MetricsPlot, self).__init__()
        # Flag to determine if there are previous data to be loaded
        self.previous_data = False
        self.epoch_counter = 0
        self.output_dir = Path(output_dir)
        self.metrics = metrics
        self.file_format = file_format

    def set_output_dir(self, output_dir):
        """Set output directory to store the plot.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path):
        """Load file containing data from previous plot.
        In case training was accidentally stopped, this function allows you
        to recover data from a previous plot and to continue plotting from it.
        """
        self.previous_data = True
        if (Path(file_path).exists()):
            self.data = pd.read_csv(file_path)
            self.epoch_counter = len(self.data)
        else:
            print('Could not read previous data. Starting plot from 0.')

    def on_train_begin(self, logs={}):
        if (not self.previous_data):
            self.data = pd.DataFrame(columns=self.metrics, dtype=float)

    def on_epoch_end(self, batch, logs={}):
        # Append metrics
        for metric in self.metrics:
            if metric in logs.keys():
                self.data.loc[self.epoch_counter, metric] = float(logs.get(metric))
        self.epoch_counter += 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.set(style='darkgrid')
        for metric in self.metrics:
            if metric in logs.keys():
                sns.lineplot(x=range(0, len(self.data)), y=metric, data=self.data, ci=None,
                             label=metric, linewidth=2)

        ax.set_xlabel("Epochs", fontsize=16)
        ax.set_ylabel("", fontsize=16)
        ax.set_title('Training Monitoring', fontsize=20)
        # Save Figure
        fig = ax.get_figure()
        fig.savefig(self.output_dir / f'TrainingMonitoring.{self.file_format}')
        # Save dataframe
        self.data.to_csv(self.output_dir / 'TrainingMonitoring.csv', index=False)
        plt.close()


# Scheduler
### The following scheduler was proposed by @marcinkaczor
### https://github.com/LIVIAETS/boundary-loss/issues/14#issuecomment-547048076
class AlphaScheduler(Callback):
    """
    Stores and update alpha parameter to weight the 2 contribution of Weighted_DiceBoundary_Loss,
    or any other function with weighted contribution.
    Args:
        alpha: parameter to weight functions. Must be in [0,1]
        update_fn: function that updates alpha every time "on_epoch_end" calls it.
    """
    def __init__(self, alpha, update_fn, progressive=True, step_epoch=40):
        self.alpha = alpha
        self.update_fn = update_fn
        self.progressive = progressive
        self.step_epoch = step_epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.progressive:
            updated_alpha = self.update_fn(K.get_value(self.alpha))
            K.set_value(self.alpha, updated_alpha)
        else:
            if (epoch % self.step_epoch) == 0 and (epoch != 0):
                updated_alpha = self.update_fn(K.get_value(self.alpha))
                K.set_value(self.alpha, updated_alpha)
