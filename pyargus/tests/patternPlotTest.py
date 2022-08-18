# -*- coding: utf-8 -*-
"""
********************************************************
*****          Demonstration and test              *****
********************************************************

PyARGUS 

Demonstration functions written to test the radPatternPlot function proper 
operation.

Tamás Pető
21, april 2017

"""
import numpy as np
from pyargus.antennaArrayPattern import array_rad_pattern_plot

def demo_ULA_plot(N=4, d=0.5, theta=90):
    """
     Description:
     ------------
         Displays the radiation pattern of a uniformly spaced linear antenna array      
    
        Parameters:
        -----------          
            N     : (int) Number of antenna elements in the antenna array. Default value: 4
            d     : (float) Distance between antenna elements. [lambda] Default value: 0.5
            theta : (float) Main beam direction. Default value: 90
    """
    # Generate weighting coefficients for main beam-steering
    w = np.ones(N,dtype=complex)

    # Main beam positioning for ULA
    for i in np.arange(0,N,1):
         w[i] = np.exp(i* 1j* 2*np.pi * d *np.cos(np.deg2rad(theta)))          
    
    # Manual override coefficients
    # W     = np.array([1,-1,1,-1],dtype = complex)
   
    # Antenna element positions for ULA
    x_coords = np.arange(N)*d - (N-1)*d/2
    y_coords = np.zeros(N)
    array_alignment = np.array((x_coords, y_coords))
    
    array_rad_pattern_plot(w=w, array_alignment = array_alignment)

def demo_UCA_plot(N=4 , r=1, theta=90):
    """
     Description:
     ------------
         Displays the radiation pattern of a uniformly spaced circular antenna array      
    
        Parameters:
        -----------          
            N     : (int) Number of antenna elements in the antenna array. Default value: 4
            r     : (float) Radius of the antenna system. [lambda] Default value: 0.5
            theta : (float) Main beam direction. Default value: 90
    """
    # Generate weighting coefficients for main beam-steering
    w = np.ones(N,dtype=complex)

    # Main beam positioning for UCA
    for i in np.arange(0,N,1):   
         w[i] = np.exp(1j*2*np.pi*r*np.cos(np.radians(theta-i*(360)/N))) # UCA   
   
    # Antenna elemnt positions for UCA
    x_coords = np.zeros(N)
    y_coords = np.zeros(N)
    for i in range(N):
        x_coords[i] = r * np.cos(i*2*np.pi/N)
        y_coords[i] = r * np.sin(i*2*np.pi/N)
    array_alignment = np.array((x_coords, y_coords))
    
    array_rad_pattern_plot(w=w, array_alignment = array_alignment)

