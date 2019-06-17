#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math


#-----------------------------------------------------------------------------------------------
# universal constants and units
h            = 6.626068e-27              # Plank constant (erg/s)
c            = 3e10                      # Spped of light (cm/s)
k_B          = 1.38064852e-16            # Boltzmann constant (erg/Kelvin)
G            = 6.6738e-8                 # Gravitational constant
Mpc_to_cm    = 3.08567758e24             # Mpc/cm
M_sun        = 1.9891e33                 # Solar mass (g)
pi           = math.pi                   # Basically pi
M_p          = 1.6726e-24                # Proton mass (g)
sig_T        = 1.640e-16

M            = 15*M_sun                  # Blackhole mass
L_edd        = 4*pi*G*M*M_p*c/sig_T      # Eddington luminosity
eta          = 0.1                       # Energy conversion efficiency
M_dot        = 1e19*(0.1/eta)*(M/M_sun)  # Mass accretion rate

r_g          = 2.*G*M/c**2               # Schwarzschild radius
r_in         = 1.83*r_g                  # Innermost radius
sigma        = 5.6704e-5                 # Stefan–Boltzmann constant (erg cm−2 s−1 K−4)

# The temperature (Kelvin) at the inner radius (cm)
# T_eff_in     = (3.*G*M_dot*M/(8*pi*r_in**3*sigma))**0.25      

p_value      = 0.63                      # Temperature dependence
distance     = 4.03*Mpc_to_cm            # Distance from M83 to the observer
deg_to_rad   = pi/180.                   # convert degree to radian
intervals    = 10000.                    # the step numbers used in the integration
step         = 100                       # step numbers for frequency range 
#-----------------------------------------------------------------------------------------------


#----------------------------------------------------------------
# load Chandra data: ULX in M83 galaxy
energy, f, f_err = np.loadtxt('plot_flux.dat', unpack='True')
#----------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------
# Slim disk model
def slim(inclination, nu, distance, p_value, pi, planck_const, k_B, r_in, r_out, T_eff_in):
	x_in   = (h*nu)/(k_B*T_eff_in)
	x_out  = (h*nu)/(k_B*T_eff_in)*(r_out/r_in)**p_value
#---------------------------------------------------------------
	# perform trapsoidal intergration
	def trap(x_in, x_out, intervals):
		#--------------------------------------------------
		def func(x):
			return x**(2./p_value - 1.)/(math.exp(x) - 1.)
		#--------------------------------------------------
		h           = (x_out - x_in)/intervals
		sum_area    = 0.0
		
		while x_in < x_out:
			part     = h*(func(x_in) + func(x_in + h))/2.
			sum_area = sum_area + part
			x_in     = x_in + h
		integration  = sum_area
		return integration	
#---------------------------------------------------------------
	trap   = trap(x_in, x_out, intervals)
	S_nu   = (math.cos(inclination)*4.*pi*h/c**2./distance**2.)*r_in**2./p_value*(k_B*T_eff_in/h/nu)**(2./p_value)*nu**3.*trap
	return S_nu
#-------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
# Calculate S_nu at different nu 
def plot_spec(inclination, color, label, r_out, p_value, T_eff_in):
	S_nu    = []

	for i in nu:
		S_nu_slim = slim(inclination, i, distance, p_value, pi, h, k_B, r_in, r_out, T_eff_in)
		S_nu.append(S_nu_slim)
	plt.plot(nu, S_nu, color=color, label=label)
#----------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------------------
# Check the frequencey dependence for p=0.5 and 0.75
def spec_dependence():

	plot_spec(0 *deg_to_rad, 'steelblue',  r'$p = 0.50; r_{out}=150r_{in}, T_{in}=1e9\ Kelvin$', 150.*r_g, 0.50, 1e9)
	plot_spec(0 *deg_to_rad, 'red',        r'$p = 0.75; r_{out}=150r_{in}, T_{in}=1e9\ Kelvin$', 150.*r_g, 0.75, 1e9)
	
	plt.plot(nu, (nu**-1)*(10**-4.5),    color='steelblue', linestyle='--')
	plt.plot(nu, (nu**(1/3))*(10**-30.95), color='red',       linestyle='--')
	
	plt.annotate(r'$S_{\nu}\propto \nu^{-1}$',          xy=(1.4*10**19, 3.26*10**-24), color='steelblue', weight='bold', size=15)
	plt.annotate(r'$S_{\nu}\propto \nu^{\frac{1}{3}}$', xy=(5.8*10**18, 3.12*10**-25), color='red',       weight='bold', size=15)
	
	plt.legend(loc='lower left')
	plt.xlabel(r'$\nu$', fontsize=18)
	plt.ylabel(r'$S_{\nu}$', fontsize=18)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

# Spectrum Examination
def run_spec_check():
	# set up frequency range 
	nu = np.logspace(17, 20.5, step)
	spec_dependence()

#run_spec_check()
#-------------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------
# Set up frequence range for simulated spectrum
step = 100                      
nu   = np.logspace(16.0, 18.3, step)

# Convert nu to keV
keV  = (nu*h/(1.6022e-9))
#-------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
# Calculate nu*S_nu at different keV based on the function slim()
def plot_fit(inclination, color, label, r_out, p_value, T_eff_in):
	nu_S_nu = []

	for i in nu:
		S_nu_slim = slim(inclination, i, distance, p_value, pi, h, k_B, r_in, r_out, T_eff_in)
		nu_S_nu.append(S_nu_slim*i)
	plt.plot(keV, nu_S_nu, color=color, label=label)

	n = len(energy)
	MSE = []

	for i in range(n):
		nu_chandra = energy[i]*(1.6022e-9)/h
		f_model    = slim(0*deg_to_rad, nu_chandra, distance, p_value, pi, h, k_B, r_in, r_out, T_eff_in)
		MSE.append((f[i] - f_model*nu_chandra)**2)
	
	MSE_val = sum(MSE)/n

	# select small MSE value
	if MSE_val < 1e-28:
		print(MSE_val)

#----------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
Temp  = np.logspace(6.68, 6.85, 5)
p     = np.linspace(0.5,  0.75, 5)
R_out = np.linspace(10.,   60.,  5)
color = ['steelblue','gold','coral','red','teal']
index = range(len(Temp))

print(energy)
def fit(index_p, index_r_out):
	for k in index:
		label = r'$p=$' + str(p[index_p]) + r'$; T_{in}=$' + str(Temp[k]) + r'$ K; R_{out}=$' + str(R_out[index_r_out]) + r'$ r_{in}$'
		plot_fit(0*deg_to_rad, color[k],  label, R_out[index_r_out]*r_g, p[index_p], Temp[k])
	
	# plot Chandra data
	kwargs = dict(ecolor='k', color='k', capsize=2, ms=7)
	plt.errorbar(energy, f, yerr=f_err, fmt='o', mfc='steelblue', **kwargs, label=r'$Chandra\ Observation$')

	plt.legend(loc='lower left')
	plt.xlabel(r'keV', fontsize=18)
	plt.ylabel(r'$\nu S_{\nu}$', fontsize=18)
	plt.xlim(0.1, 10.0)
	plt.ylim(1e-17, 1e-12)
	
	plt.xscale('log')
	plt.yscale('log')

def run_fit():
	for i in index:
		for j in index:
			print(i, j)
			fit(i, j)
			#plt.savefig('fig/'+str(i)+'_'+str(j)+'.png', dpi=150)
			plt.show()

# Run the fitting
#run_fit()

# Plot best fit
def best_fit():
	label = r'$p=$' + str(p[2]) + r'$; T_{in}=$' + str(Temp[1]) + r'$ K; R_{out}=$' + str(R_out[0]) + r'$ r_{in}$'
	plot_fit(0*deg_to_rad, 'teal',  label, R_out[0]*r_g, p[2], Temp[1])
	
	# plot Chandra data
	kwargs = dict(ecolor='k', color='k', capsize=2, ms=7)
	plt.errorbar(energy, f, yerr=f_err, fmt='o', mfc='steelblue', **kwargs, label=r'$Chandra\ Observation$')
	
	plt.legend(loc='lower left')
	plt.xlabel(r'keV', fontsize=18)
	plt.ylabel(r'$\nu S_{\nu}$', fontsize=18)
	plt.xlim(0.1, 10.0)
	plt.ylim(1e-17, 1e-12)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
# Plot the best fitted simulated spectra (set up the index by myself)
#best_fit()
#-----------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
# set up the temperature range based on Chandra spectral fitting
inclination=0.0
cos_theta = math.cos(inclination*deg_to_rad)
temp = [5990719.989183, 4867531.939266, 7113908.0391]
norm =  0.0401643   #[0.0401643, 0.0728194]
D10 = 4.03*10**3/10
Rin_obs = (norm/cos_theta)**0.5*D10*10**5
#------------------------------------------------------------------------------------------------------------

