"""Projet SDAS : simulation de la vitesse v(x) d'un véhicule sur un tronçon.

Grâce à ce fichier on peut écrire :  from vitesse import Noise, SpeedPolynomial"""
from vitesse.measurements import create_rng, generate_positions, simulate_speeds
from vitesse.noise import Noise
from vitesse.polynomial import SpeedPolynomial
from vitesse.maximum_likelihood import MaximumLikelihood, log_likelihood, log_likelihood_variable_sigma

__all__ = ["Noise", "SpeedPolynomial", "create_rng", "generate_positions", "simulate_speeds",
           "MaximumLikelihood", "log_likelihood", "log_likelihood_variable_sigma"]
