import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import keras
from keras.preprocessing.sequence import Sequence
import random


metrics_short_ref = {'loss': 'Loss', 
                     'categorical_accuracy': 'Accuracy'
                    }


def load_dataset(path, exclude_record=None, channels_ref=None, verbose=True):
    """Import HDF5 databases from path.
    path: path of folder containing HDF5 databases
    exclude_record: list of databases to exclude
    channels_ref: list of channels to include
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

            if channels_ref is None:
                channels_ref = list(f.keys())

            if verbose:
                print(record_id)
                
            for idx_channel, channel in enumerate(channels_ref):
              
                data = f[channel][::]
                data = data[:, np.newaxis, :]

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


def to_input(u, tensor=False):
    """Convert data array to shape (batch, data).
    u: array with shape (batch, samples, channels)
    
    returns x_out: array x with shape (batch, samples*selected_channels) if tensor is False
                   array x with shape (batch, samples, selected_channels, 1) if tensor is True"""
    
    if tensor:
        u_out = u.reshape((u.shape[0], u.shape[1], u.shape[2], 1))
        
    else:
        u_out = u.reshape((u.shape[0], u.shape[1]*u.shape[2]))
    
    return u_out


def to_output(u, num_classes=5):
    """Convert label array to one-hot-encoding with shape (batch, binary_label).
    u: label array with shape (batch, 1)
    
    returns: u_out (array with shape (batch, binary_label))"""

    u_out = keras.utils.to_categorical(u, num_classes=num_classes)
    
    return u_out
    
  
def cross_validation(dataset, model_ref,
                     tensor=False, epochs=10,  batch_size=128, 
                     plot=True, verbose=True):
    """Leave-one-out cross validation scheme
    dataset: dataset containing records with PSG data and hypnograms
    model_ref: Keras model
    tensor: provide tensor shape PSG data
    epochs: number of training epochs
    batch_size : number of mini-batch samples
    verbose: print intermediate results
    
    returns metrics: list with train and test accuracy"""
    
    title = 'model ' + model_ref().name + ' - ' \
            + str(len(dataset)) + ' fold cross-validation'
    
    print(title, '\n')
    
    dataset_records = list(dataset.keys())
    
    stats = []
    
    for idx_record, record in enumerate(dataset_records):
      
        records_train = [record]
        records_test = [key for key in dataset_records 
                            if key not in records_train]
        
        generator_train = PSGSequence(dataset, 
                                      records=records_train, 
                                      batch_size=batch_size,
                                      tensor=tensor)

        generator_test = PSGSequence(dataset, 
                                     records=records_test,
                                     batch_size=1024,
                                     tensor=tensor)
        
        x_shape = generator_train.x_batch_shape[1:]
        y_shape = generator_train.y_batch_shape[1:]

        model = model_ref(x_shape, y_shape)
        
        stats_run_init_train = model.evaluate_generator(generator_train)
        stats_run_init_test = model.evaluate_generator(generator_test)
        
        stats_run = {}
        for idx_metric, metric in enumerate(model.metrics_names):
            
            stats_run[metric] = [stats_run_init_train[idx_metric]]
            stats_run['val_'+metric] = [stats_run_init_test[idx_metric]]
          
        history = model.fit_generator(generator_train, 
                                      epochs=epochs, 
                                      validation_data=generator_test,
                                      verbose=False)

        for idx_metric, metric in enumerate(model.metrics_names):

            stats_run[metric] += history.history[metric]
            stats_run['val_'+metric] += history.history['val_'+metric]

        stats += [stats_run]

    if plot:
      
        plot_cross_validation(stats, model_ref().metrics_names, title=title)
        
    return stats
  
  
def plot_cross_validation(stats, metrics_names,
                          metrics_short=metrics_short_ref,
                          title=None, ylim=None, 
                          fig_w=4.5, fig_h=3.5):

    fig, axs = plt.subplots(1, 2, figsize=(2*fig_w, fig_h))

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.99)

    for idx_metric, metric in enumerate(metrics_names):
      
        if metric not in metrics_short:
          
            metrics_short[metric] = metric
            
        data = np.array([item[metric] for item in stats])
        data_val = np.array([item['val_'+metric] for item in stats])

        if idx_metric==0:
            labels = ['train', 'val']
        else:
            labels = [None, None]



        plt.sca(axs[idx_metric])
        plt.plot(data[0], 'C0', alpha=0.1)
        plt.plot(data[1:].T, 'C0', alpha=0.1)
        plt.plot(np.mean(data, axis=0), 'C0', linewidth=2, label=labels[0])

        plt.plot(data_val[0], 'C1', alpha=0.1)
        plt.plot(data_val[1:].T, 'C1', alpha=0.1)
        plt.plot(np.mean(data_val, axis=0), 'C1', linewidth=2, label=labels[1])

        plt.title(metrics_short[metric])
        plt.ylabel(metrics_short[metric])
        plt.xlabel('epochs')

        if idx_metric==0:
            plt.legend()

    plt.tight_layout()
    fig.canvas.draw()
    plt.show()
    
    return fig


def plot_dataset_stats(dataset, exclude_record=None, title=None,
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

    fig, axs = plt.subplots(1, n_channels, figsize=(6*n_channels,3))
    
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.05)
        
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
