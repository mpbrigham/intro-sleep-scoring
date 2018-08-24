import numpy as np
import os
import h5py


def load_dataset(path, channels=None, exclude=None, verbose=True):
    """Import HDF5 databases from path.
    path: path of folder containing HDF5 databases
    channels: list of channels to include
    exclude: list of databases to exclude
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
                
            for channel_idx, channel in enumerate(channels):
              
                data = f[channel][::]
                data = data[:, np.newaxis, :]
                
                if channel_idx==0:
                    x = data
                else:
                    x = np.concatenate((x, data), axis=1)

            y = np.array([f['stages'][::]]).T

            dataset[record_id] = {'path': record_path,
                                  'channels': channels,
                                  'samples': x,
                                  'hypnogram': y
                                 }
            
    if verbose:
        print('\nImported', len(dataset), 'records')
        
    return dataset


def plot_stats(dataset, x_min=-1, x_max=1, x_n=5e4):
    """Plot per channel histograms of dataset.
    dataset: dictionary with dataset, PSG channels with shape (batch, channel, data).
    """ 

    x_bins = np.linspace(x_min, x_max, num=100)
    n_channels = len(channels)

    fig, axs = plt.subplots(1, n_channels, figsize=(6*n_channels,3))

    for key_idx, key in enumerate(dataset):

        for channel_idx, channel in enumerate(channels):
            data = dataset[key]['samples'][:,channel_idx].flatten()
            if data.shape[0]>=x_n:
                data = np.random.choice(data, size=int(x_n), replace=False)
            
            if channel_idx==0:
                label=key
            else:
                label=None            

            plt.sca(axs[channel_idx])           
            plt.hist(data, bins=x_bins, density=True, alpha=0.4, label=label)

            if key_idx==0:
                plt.title('Histogram '+channel)

    plt.sca(axs[0])
    plt.ylabel('Density')
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
