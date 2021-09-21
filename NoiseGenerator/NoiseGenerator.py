import numpy as np 
import pandas as pd 
import scipy.fft 
import scipy.signal
import sys 
import matplotlib.pyplot as plt  
from scipy.interpolate import interp1d

import cmath #complex exponentials

plt.style.use("~/evanstyle.mplstyle")

class NoiseGenerator:
	#initialize with a power spectral density file for various peaking times
	def __init__(self, spectrum_file, peaking_time=1.2):
		self.infile = spectrum_file
		self.pt = peaking_time 
		self.noise_spectrum_df = None



	def set_peaking_time(self, pt):
		available_pts = [0.6, 1.2, 2.4, 3.6]
		if(pt not in available_pts):
			print("Please set an available peaking time: " , end = ' ')
			print(available_pts)
		self.pt = pt 


	def load_noise_spectrum(self):
		#noise spectra in V^2/Hz, freq in Hz, for ***gain x6 setting*** and different peaking times
		self.noise_spectrum_df = pd.read_csv(self.infile, delimiter=',', header=0, names=["freq", "0.6", "1.2", "2.4", "3.6"], skiprows=2)


	def plot_noise_spectrum(self):
		#plot noise spectrum
		fig, ax = plt.subplots(figsize=(12, 8))
		ax.plot(self.noise_spectrum_df["freq"], self.noise_spectrum_df["0.6"], label="0.6 us tp", linewidth=3)
		ax.plot(self.noise_spectrum_df["freq"], self.noise_spectrum_df["1.2"], label="1.2 us tp", linewidth=3)
		ax.plot(self.noise_spectrum_df["freq"], self.noise_spectrum_df["2.4"], label="2.4 us tp", linewidth=3)
		ax.plot(self.noise_spectrum_df["freq"], self.noise_spectrum_df["3.6"], label="3.6 us tp", linewidth=3)
		ax.set_xlabel("freq (Hz)")
		ax.set_xscale('log')
		ax.set_ylabel("V^2/Hz")
		ax.set_yscale('log')
		ax.legend(loc='lower left', fancybox=True, framealpha=1)
		plt.show()

	#takes the noise spectral density and generates a random
	#time series of data using the following method:
	#(1): interpolate the frequency spectrum so that the 
	#number of samples and frequency spacing matches the fourier
	#dual for the desired sampling rate and time window in time domain. 
	#this also requires extrapolating to 0 Hz, and mirroring the PSD to 
	#have negative frequencies (symmetric) and an even number of samples total. 
	#(2): Convert power to amplitude (go from V^2/Hz to amp V/sqrt(Hz))
	#(3): For each frequency, generate a uniform-random phase between 0 and 2pi
	#(4): Frequency domain signal is now A(k)*e^(ixphi) where A is amplitude and 
	#phi is phase for that frequency
	#(5): inverse fourier transform with the random phase information to get the time domain signal. 

	#sampling rate in Hz
	def sample_random_waveform(self, sampling_rate, n_samples, debug=0):
		#time domain parameters
		sampling_rate = float(sampling_rate) #make division more certain
		sampling_period = 1.0/sampling_rate
		total_time = sampling_period*n_samples 

		#frequency domain parameters
		dF = 1.0/total_time #spacing between frequency points, the frequency resolution. 
		f_high = sampling_rate #highest frequency in the PSD for time series of sampling rate
		#form a series of positive and negative frequencies, so that the resulting
		#PSD is symmetric about 0 
		new_freqs_p = list(np.arange(0, f_high + dF, dF))
		new_freqs_n = sorted([-1*_ for _ in new_freqs_p])[:-1] #no 0 in this list, perfectly symmetric

		#get the PSD extrapolated/interpolated for these frequencies.
		#***Todo: this is introducing a small error, that is visible
		#when plotting the interpolated PSD on a log scale. The interpolation
		#error creates humps in the PSD that propagate to the generated random waveform. 
		#I think we can fix this by not interpolating, and then rescaling the resolution somehow. 
		data_freqs = self.noise_spectrum_df["freq"]
		power_interp = interp1d(data_freqs, self.noise_spectrum_df[str(self.pt)], kind='linear', fill_value='extrapolate', bounds_error=False)
		new_power_p = power_interp(np.array(new_freqs_p))
		new_power_n = (new_power_p[::-1])[:-1] #reverse order, and remove the point at 0 Hz
		#add both sides of freq and power spectrum to one list
		new_power = list(new_power_n) + list(new_power_p)
		new_freqs = list(new_freqs_n) + list(new_freqs_p)


		#debugging plot to see what the freq spectrum should be
		#NOTE, bumps will appear as a wierd plotting artifact that
		#goes away if y-scale is linear. 
		if(debug):
			fig, ax = plt.subplots(figsize=(12, 8))
			ax.plot(self.noise_spectrum_df["freq"], self.noise_spectrum_df[str(self.pt)], 'o--',markersize=4,label=str(self.pt) + " unranged", linewidth=3)
			ax.plot(new_freqs, new_power, 'o--', markersize=1, label=str(self.pt) + " ranged", linewidth=3)
			ax.set_xlabel("freq (Hz)")
			#ax.set_xscale('log')
			ax.set_xlim([-2e6, 2e6])
			ax.set_ylabel("V^2/Hz")
			ax.set_yscale('log')
			ax.legend(loc='lower left', fancybox=True, framealpha=1)
			plt.show()
		

		#convert power density to amplitude density
		#multiplying by the frequency range for units of Volts*sqrt(Hz). 
		#The other normalizations here, len(new_power) and dF/2.0 are the hardest
		#to pin down in literature. However, the factors as they are inserted
		#here result in a matching power spectral density of the final waveform. 
		new_amplitudes = [len(new_power)*np.sqrt(_*dF/2.0) for _ in new_power] 

		#create phasors with random phases
		phases = np.random.uniform(0, 2*np.pi, len(new_freqs)) #one for each frequency
		phasors = [new_amplitudes[i]*(cmath.exp(1j*phases[i])) for i in range(len(new_freqs))]

		#inverse fourier transform the phasors
		time_series = scipy.fft.ifft(phasors, n_samples)
		time_series = time_series.real
		times = np.arange(0, n_samples/sampling_rate, sampling_period)
		#debug plotting
		if(debug):
			fig, ax = plt.subplots(figsize=(12,16), nrows=2)
			ax[0].plot(times, time_series)
			ax[1].plot(new_freqs, new_power, 'o', label="input fft", linewidth=3)
			fs, pxx = scipy.signal.periodogram(time_series, sampling_rate, scaling='density')
			print(pxx[50]/new_power_p[50])
			ax[1].plot(fs[1:], pxx[1:], 'o', label="output FFT", linewidth=3)
			ax[1].set_xlabel("freq (Hz)")
			#ax[1].set_xscale('log')
			ax[1].set_ylabel("V^2/Hz")
			ax[1].set_yscale('log')
			ax[1].legend()

			plt.show()

		return times, time_series #waveform in V, times in seconds




