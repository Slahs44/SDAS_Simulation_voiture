"""Projet SDAS : simulation de la vitesse v(x) d'un véhicule sur un tronçon.

Grâce à ce fichier on peut écrire :  from vitesse import Noise, SpeedPolynomial"""
from vitesse.measurements import generate_positions, simulate_speeds
from vitesse.noise import Noise
from vitesse.polynomial import SpeedPolynomial

__all__ = ["Noise", "SpeedPolynomial", "generate_positions", "simulate_speeds"]
