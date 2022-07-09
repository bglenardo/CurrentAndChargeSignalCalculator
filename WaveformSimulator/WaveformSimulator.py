import numpy as np
from SignalCalculator import SignalCalculator

def MakeWaveform(charge_energy,peak_loc,num_samples=3500,\
                 noise_sigma=50,baseline_samples=500,\
                 offset_counts=0,drift_counts=0,\
                 electron_lifetime=200.,inf_EL=False,
                 v_drift=1.7,polarity=1,preamp_decay=500.,\
                 sampling_period=40.,grids=None):
    """
    Generate a fake waveform with specified location, energy, and noise parameters

    Arguments:
    charge_energy     :    total charge energy deposited [ADC counts]
    peak_loc          :    drift time (z-position) of event [us]
    num_samples       :    number of samples in waveform
    noise_sigma       :    standard deviation of Gaussian white noise added to waveform
    baseline_samples  :    number of samples before waveform starts
    offset_counts     :    offset added to the waveform [ADC counts]
    drift_counts      :    drift over the length of the waveform [ADC counts]
    electron_lifetime :    electron lifetime in the detector [us]
    inf_EL            :    (bool) infinite electron liftime. No z-dependence on charge energy
    v_drift           :    drift velocity [mm/us]
    polarity          :    -1 if raw waveform is flipped
    preamp_decay      :    time constant of charge preamplifier [us]
    sampling_period   :    digitizer sampling period [ns]
    grids             :    (tuple) output of MakeInterpolationGrids. If this argument is
                           present, waveform will be interpolated from existing grid rather
                           than recalculated
    
    Returns:
    waveform as a numpy array of length num_samples
    """
    samples = np.linspace(0,num_samples-1,num_samples)
    det_energy = charge_energy*max((np.exp(-peak_loc/electron_lifetime),inf_EL))
    wvfm = np.zeros(len(samples))
    peak_samples = baseline_samples + int(round(peak_loc*1000/sampling_period))
    decay_samples = int(round(preamp_decay*1000/sampling_period))
    if grids is not None:
        charge_grid,drift_grid,charge_e,drift_t = grids
        drift_pts,charge_pts = InterpolateWaveform(det_energy,peak_loc,charge_grid,\
                                                     drift_grid,charge_e,drift_t)
        drift_points = np.linspace(min(drift_pts),max(drift_pts),peak_samples-baseline_samples)
        charge_wfm = np.interp(drift_points,drift_pts,charge_pts)
    else:
        drift_points,charge_wfm = SignalCalculator.ComputeChargeWaveformOnStripWithIons( \
                                                    det_energy, 1.5, 0, (peak_samples-baseline_samples) \
                                                    *sampling_period*v_drift/1000., \
                                                    padSize=3., numPads=30, \
                                                    numWfmPoints=peak_samples-baseline_samples )
    wvfm[baseline_samples:peak_samples] = charge_wfm
    wvfm[peak_samples:] = max(wvfm)*np.exp(-(samples[peak_samples:]-peak_samples)/(decay_samples))
    noise = np.random.normal(0,noise_sigma,len(samples))
    wvfm = wvfm + noise + samples*drift_counts/len(samples) + offset_counts
    wvfm = wvfm*polarity
    drft = (samples-baseline_samples)*sampling_period/1000.
    return drft,wvfm

def MakeInterpolationGrids(num_charge_pts,num_drift_pts,\
                          charge_min=0.,charge_max=1000.,\
                          drift_min=0.,drift_max=80.,wfm_length=1000,\
                          v_drift=1.7):
    """
    Make the grids of calculated waveforms from which waveforms of arbitrary
    charge energy and drift time can be interpolated
    
    Arguments:
    num_charge_pts    :    number of charge values used to make grid
    num_drift_pts     :    number of drift time values used to make grid
    charge_min        :    minimum charge energy [ADC counts]
    charge_max        :    maximum charge energy [ADC counts]
    drift_min         :    minimum drift time [us]
    drift_max         :    maximum drift time [us]
    wfm_length        :    number of points used in signal calculator function
    v_drift           :    drift velocity [mm/us]
    
    Returns:
    charge_grid  :    num_charge_pts x num_drift_pts x wfm_length numpy array of
                      charge energy values
    drift_grid   :    num_charge_pts x num_drift_pts x wfm_length numpy array of
                      drift time values
    charge_array :    numpy array of charge values where waveforms were calculated
    drift_array  :    numpy array of drift times where waveforms were calculated
    """
    charge_array = np.linspace(charge_min,charge_max,num_charge_pts)
    drift_array = np.linspace(drift_min,drift_max,num_drift_pts)
    charge_grid = []
    drift_grid = []
    for i in range(num_charge_pts):
        charge_row = []
        drift_row = []
        for j in range(num_drift_pts):
            drift_points,charge_wfm = SignalCalculator.ComputeChargeWaveformOnStripWithIons( \
                                                    charge_array[i], 1.5, 0, drift_array[j]*v_drift, \
                                                    padSize=3., numPads=30, \
                                                    numWfmPoints=wfm_length )
            charge_row.append(charge_wfm)
            drift_row.append(drift_points)
        charge_grid.append(charge_row)
        drift_grid.append(drift_row)
    return np.array(charge_grid),np.array(drift_grid),charge_array,drift_array

def InterpolateWaveform(charge,drift,charge_grid,drift_grid,\
                        charge_energy,drift_time,num_points=1000):
    """
    Uses grids and arrays from MakeInterpolationGrids() to interpolate waveforms
    at arbitrary charge energy and drift times. Starts by finding the peak of the
    four nearest charge waveforms (charge energy and drift times below and above
    point of interest). Does interpolation in drift time first, computing an upper
    and lower waveform at the given drift time and the charge energies above and
    below the energy of interest. Finished by interpolating upper and lower waveforms
    to get waveform at charge energy of interest
    
    Arguments:
    charge         :    charge energy [ADC counts]
    drift          :    drift time [us]
    charge_grid,
    drift_grid,
    charge_energy,
    drift_time     :    outputs from MakeInterpolationGrids()
    num_points     :    number of points used in interpolation
    
    Returns:
    numpy array of drift times
    numpy array of corresponding charge energies
    """
    charge_ind = np.argwhere((charge>charge_energy)==1)[-1]
    drift_ind = np.argwhere((drift>drift_time)==1)[-1]
    charge_diff = (charge-charge_energy[charge_ind])/(charge_energy[charge_ind+1]-charge_energy[charge_ind])
    drift_diff = (drift-drift_time[drift_ind])/(drift_time[drift_ind+1]-drift_time[drift_ind])
    charge_x_l = np.linspace(min(charge_grid[charge_ind,drift_ind][0,:]),\
                             max(charge_grid[charge_ind,drift_ind][0,:]),num_points)
    charge_x_h = np.linspace(min(charge_grid[charge_ind+1,drift_ind][0,:]),\
                             max(charge_grid[charge_ind+1,drift_ind][0,:]),num_points)
    drift_f_l_l = np.interp(charge_x_l,charge_grid[charge_ind,drift_ind][0,:],\
                            drift_grid[charge_ind,drift_ind][0,:])
    drift_f_l_h = np.interp(charge_x_h,charge_grid[charge_ind+1,drift_ind][0,:],\
                            drift_grid[charge_ind+1,drift_ind][0,:])
    drift_f_h_l = np.interp(charge_x_l,charge_grid[charge_ind,drift_ind+1][0,:],\
                            drift_grid[charge_ind,drift_ind+1][0,:])
    drift_f_h_h = np.interp(charge_x_h,charge_grid[charge_ind+1,drift_ind+1][0,:],\
                            drift_grid[charge_ind+1,drift_ind+1][0,:])
    drift_f_l = drift_f_l_l + drift_diff*(drift_f_h_l - drift_f_l_l)
    drift_f_h = drift_f_l_h + drift_diff*(drift_f_h_h - drift_f_l_h)
    drift_x = np.linspace(min(drift_f_l),max(drift_f_l),num_points)
    charge_f_l = np.interp(drift_x,drift_f_l,charge_x_l)
    charge_f_h = np.interp(drift_x,drift_f_h,charge_x_h)
    return drift_x,charge_f_l + charge_diff*(charge_f_h - charge_f_l)
