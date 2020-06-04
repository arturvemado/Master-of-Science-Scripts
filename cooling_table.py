# Author: Artur Vemado (artur.vemado@usp.br)
# Code to generate a radiative cooling following the equations
# Narayan & Yi 1995
# Radiative cooling in function of R, Te, ne (radius, electron temperature
# and electron number density) in CGS.

# ESSENTIAL PARAMETERS:
# C_Mbh, rho_0, r_0, r_min, beta

# Change section "Defining range of ne, R, Te" to generate a table with the size
# and ranges of your interests

from scipy.special import kn
from scipy.optimize import root
import numpy as np

# Constants to be used in cgs
C_sigma = 5.67051e-5		#Stephan Boltzmann constant
C_kb = 1.3806505e-16		#Boltzmann constant
C_h = 6.62606876e-27		#Planck constant
C_me = 9.1093826e-28		#Electron mass
C_mp = 1.67262171e-24		#Proton mass
C_amu = 1.66053886e-24		#Atomic mass unit
C_c = 2.99792458e10		#Speed of light
C_G = 6.6726e-8			#Gravitational constant
C_Msun = 2.e33			#Sun mass
C_sigmaT = 6.6524e-25		#Thomson cross section
#C_pi = 3.14159265358979	#PI

#Mass of the Black hole
C_Mbh = 10. * C_Msun
C_GM = C_Mbh * C_G

#Schwarzschild Radius
Rs = 2. * C_G * C_Mbh / (C_c * C_c)

#Adiabatic index
C_gamma = 5./3.

#Constant of temperature parametrization
CONST_1 = C_kb / (C_me * C_c * C_c)

#Temperature parametrization
def theta_e (Te):
	return (CONST_1 * Te)

#Disk geometry and polytropic constant
rho_0 = 5e-7				#Maximum density of initial condition
r_0 = 100. * Rs 			#Radius of maximum density (rho_0)
r_min = 75. * Rs 			#Minimum raius of torus
CONST_2 = - C_GM/(r_min-Rs) + C_GM/(2.*r_min*r_min)*(r_0*r_0*r_0)/((r_0-Rs)*(r_0-Rs))
kappa = (C_gamma-1.)/C_gamma*pow(rho_0, 1.-C_gamma)*(CONST_2 + C_GM/(r_0-Rs) 
		- C_GM/2. * r_0/((r_0-Rs)*(r_0-Rs)))								#Polytropic constant

#Entangled magnetic field (local randomly oriented magnetic field)
beta = 10.

#Sound speed cs = sqrt(d P/d rho)
def sound_speed(ne):
	result = C_gamma * kappa * pow(ne * 1.14 * C_amu, C_gamma-1.)
	result = np.sqrt(result)
	return result

#Magnetic field assuming equipartition
def B(ne):
	result = 8.*np.pi*sound_speed(ne)*sound_speed(ne)*ne*1.14*C_amu/(beta+1.)
	result = np.sqrt(result)
	return result

#Scale Height following the expression cs/omega_K
def scale_height(R, ne):
	#result = np.sqrt(R/C_GM)*sound_speed(ne)*(R-Rs)
	return R

#BREEMSTRAHLUNG PART
#Electron-ion collision
def Fei(Te):
	th_e = theta_e(Te)

	if th_e >= 1.:
		result = 9.*th_e/(2.*np.pi)
		result *=(np.log(1.123*th_e+0.48)+1.5)
	else:
		result = 4.*np.sqrt(2.*th_e/(np.pi**3.))
		result *= (1.+1.781*pow(th_e, 1.34))

	return result

#Electron-ion colling rate
def Qei(ne, Te):
	result = 1.48e-22
	result *= (ne*ne*Fei(Te))

	return result

#Electron-Electron collision
def Qee(ne, Te):

	th_e = theta_e(Te)

	if th_e <= 1.:
		result = 2.56e-22
		result *= (ne*ne*pow(th_e, 1.5))
		result *= (1.+1.1*th_e+th_e*th_e-1.25*pow(th_e, 2.5))
	else:
		result = 3.4e-22
		result *= (ne*ne*th_e)
		result *= (np.log(1.123*th_e)+1.28)
	return result

#Breemstralung cooling rate
def Qbrem(ne, Te):
	result = Qei(ne, Te) + Qee(ne, Te)
	return result


#SYNCHROTRON PART
#Function for simplicity, using bessel function
def BTB(Te, ne):
	result = kn(2,1./theta_e(Te)) * theta_e(Te) * theta_e(Te) * theta_e(Te) * B(ne)
	result = 1. / result
	return result

#Transcedental equation for xm
def TransEq(x, Te, R, ne):
	result = 1./x**(7./6.) + 0.4/x**(17./12.) + 0.5316/x**(5./3.)
	result *= 2.49e-10*12*np.pi*ne*scale_height(R, ne)*BTB(Te, ne)	
	result -= np.exp(1.8899*x**(1./3.))
	return result

#critical frequency
def nu_c(ne, Te, R):
	xm = root(TransEq, 1.e-3, args=(Te, R, ne))
	#print(xm.x)
	result = 3. * 2.8e6 * B(ne) * theta_e(Te) * theta_e(Te) * xm.x[0]/2.
	return result

#Synchrotron cooling rate
def Qsyn(ne, R, Te):
	result = nu_c(ne, Te, R)
	result = result * result * result
	result = 2. * np.pi * C_kb * Te * result
	result = result/(3. * C_c * C_c * R)
	return result


#SYNCHROTRON SELF COMPTON PART
#Scattering optical depth
def tau_es(R, ne):
	result = 2. * ne * C_sigmaT * scale_height(R, ne)
	return result

#Mean amplification factor in the energy of the scattered photon when scattering electrons 
#have a Maxwellian velocity distribution of temperature
def Amp(Te):
	return (1. + 4.*theta_e(Te) + 16.*theta_e(Te)*theta_e(Te))

#Energy normalization
def enorm(ne, Te, R):
	result = C_h * nu_c(ne, Te, R)
	result = result/(C_me * C_c * C_c)
	return result

#Probability of scattering a photon
def Prob(R, ne):
	return 1. - np.exp(-tau_es(R, ne))

#Comptonized energy anhancement factor
def eta(ne, R, Te):
	eta1 = Prob(R, ne) * (Amp(Te)-1.)
	eta1 = eta1 / (1. - Prob(R, ne)*Amp(Te))
	eta3 = -1. - np.log(Prob(R, ne))/np.log(Amp(Te))
	result = 1. + eta1 - eta1*(enorm(ne, Te, R)/(3.*theta_e(Te)))**(eta3)
	return result

#Synchrotron self compton cooling rate
def Qssc(ne, R, Te):
	result = Qsyn(ne, R, Te)*(eta(ne, R, Te) - 1.)
	return result

#Total cooling rate in optically thin approximation
def Q1(ne, R, Te):
	result = Qbrem(ne, Te) + Qsyn(ne, R, Te) + Qssc(ne, R, Te)
	return result

#Absorption optical depth
def tau_abs(ne, R, Te):
	result = scale_height(R, ne) * Q1(ne, R, Te)
	result = result/(4.*C_sigma*Te*Te*Te*Te)
	return result

#total optical depth in the vertical direction from the disk midplane surface
def tau_tot(ne, R, Te):
	result = tau_abs(ne, R, Te) + tau_es(R, ne)
	return result

#Resulting cooling rate for both optically thick and optically thin cooling limits
def Qtot(ne, R, Te):
	result = 4. * C_sigma * Te * Te * Te * Te / scale_height(R, ne)
	result = result / (3.*tau_tot(ne, R, Te)/2. + np.sqrt(3.) + 1./tau_abs(ne, R, Te))
	return result

'''
#Defining range of ne, R, Te
num_div = 100
Te = np.logspace(6, 12, num=num_div)
ne = np.logspace(12, 20.3, num=num_div)
R = np.logspace(np.log10(1.3*Rs), np.log10(400*Rs), num=num_div)


#CALCULATING AND GENERATING COOLING TABLE
#Creating file
f_cool = open("coolingtable.dat", "w+")
f_ne = open("eletronicdensity.dat", "w+")
f_Te = open("temperature.dat", "w+")
f_R = open("radius.dat", "w+")

#Writing to file
for i in range(len(ne)):
	f_ne.write("%e\n" % (ne[i]))
	for j in range(len(R)):
		if i==0:
			f_R.write("%e\n" % (R[j]))
		for k in range(len(Te)):
			if i==0 and j==0:
				f_Te.write("%e\n" % (Te[k]))
			f_cool.write("%e\n" % (Qtot(ne[i], R[j], Te[k])))


#Closing files
f_cool.close()
f_ne.close()
f_Te.close()
f_R.close()

#THE END - ENJOY YOUR COOLING
'''
















