import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from IPython.core.display import HTML

def load_dataset(path, exclude_record=None, channels_ref=None, remove_mean=False, verbose=True):
    """Import HDF5 databases from path.
    path: path of folder containing HDF5 databases
    exclude_record: list of databases to exclude
    channels_ref: list of channels to include
    remove_mean: remove mean channel value
    verbose: extensive reporting

    returns dataset: dictionary with contents of HDF5 databases in each key,
        where key name is the filename without file extension. Each key contains
        a dictionary with the following keys:
            'path' with the filesystem path of the database
            'channels' with the list of channels
            'samples' with the PSG data ordered to 'channels' in axis 0
            'hypnogram' with the hypnogram
    """

    records = [record for record in os.listdir(path)
                   if os.path.splitext(record)[1].lower()=='.h5']

    dataset = {}

    if verbose:
        print('processing', len(records), 'records in folder', path, '\n')
        
    for record in records:

        record_id = os.path.splitext(record)[0].lower()
        record_path = os.path.join(path, record)

        with h5py.File(record_path, 'r') as f:

            if verbose:
                print(record_id)
                
            for idx_channel, channel in enumerate(channels_ref):
              
                data = f[channel][::]
                data = data[:, np.newaxis, :]
                
                if remove_mean:
                    data = data - np.mean(data)

                if idx_channel==0:
                    x = data
                else:
                    x = np.concatenate((x, data), axis=1)

            y = np.array([f['stages'][::]]).T

            dataset[record_id] = {'path': record_path,
                                  'channels_ref': channels_ref,
                                  'channels': x,
                                  'hypnogram': y
                                 }
            
    if verbose:
        print('\nImported', len(dataset), 'records')
        
    return dataset


def plot_stats(dataset, exclude_record=None, title=None,
        x_min=-1, x_max=1, y_max=None, x_n=5e4):
    """Plot per channel histograms of PSG record in dataset.
    dataset: dataset dictionary with PSG channels with shape (batch, channel, data)
    exclude_record: list of records to exclude
    Title: global title
    x_min: x-axis minimum value
    x_max: x-axis maximum value
    x_n: number of samples for histogram
    """

    dataset_keys = list(dataset.keys())
    if exclude_record is not None:
        dataset_keys = [key for key in dataset_keys
                            if key not in exclude_record]

    channels_ref = dataset[dataset_keys[0]]['channels_ref']

    n_channels = len(channels_ref)
    
    x_bins = np.linspace(x_min, x_max, num=100)

    if title is not None:
        display(HTML('<h2>'+title+'</h2>'))

    fig, axs = plt.subplots(1, n_channels, figsize=(6*n_channels,3))

    for idx_record_id, record_id in enumerate(dataset_keys):

        for idx_channel, channel in enumerate(channels_ref):

            data = dataset[record_id]['channels'][:,idx_channel].flatten()

            if data.shape[0]>=x_n:
                data = np.random.choice(data, size=int(x_n), replace=False)
            
            if idx_channel==0:
                label=record_id
            else:
                label=None            

            plt.sca(axs[idx_channel])           
            plt.hist(data, bins=x_bins, density=True, alpha=0.4, label=label)

            if idx_record_id==0:
                plt.title('Histogram '+channel)
                if y_max is not None:
                    plt.ylim([0,y_max])

    plt.sca(axs[0])
    plt.ylabel('Density')
    plt.xlabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    fig.canvas.draw()


def print_stats(x, name='EEG'):
    """Print minimum, maximum, mean and standard deviation 
    along dimension 0 of array.
    x: array with shape (batch, channel, data)
    """

    print(name+'\t min\t\t max\t\t mean\t\t std\n')

    for idx in range(x.shape[1]):
        print(idx, 
            '\t', '{0:.4f}'.format(np.min(x[:,idx])), 
            '\t', '{0:.4f}'.format(np.max(x[:,idx])), 
            '\t', '{0:.4f}'.format(np.mean(x[:,idx])), 
            '\t', '{0:.4f}'.format(np.std(x[:,idx])))