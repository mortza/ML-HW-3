#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16  01:49:49 2017

@author: mortza
"""
from collections import namedtuple
import numpy as np


class EM:

    def __init__(self, k, convergence=1e-4, max_iters=1000):
        self.n_centroids = k
        self.convergence = convergence
        self.max_iters = max_iters
        self.parameters = namedtuple(
            'parameters', ['mu', 'sigma', 'w', 'log_likelihoods', 'num_iters'])

    def P(self, mu, sigma, X):
        t1 = np.linalg.det(sigma) ** -.5 * \
            (2 * np.pi) ** (-self.n_features / 2.)
        t2 = np.dot(np.linalg.inv(sigma), (X - mu).T).T
        t3 = np.exp(-.5 * np.einsum('ij, ij -> i', X - mu, t2))
        return t1 * t3

    def P_predict(self, mu, sigma, x):
        t1 = np.linalg.det(sigma) ** -.5 * \
            (2 * np.pi) ** (-self.n_features / 2.)
        t2 = np.dot(np.linalg.inv(sigma), (x - mu).T).T
        t3 = np.exp(-.5 * np.dot(x - mu, t2))
        return t1 * t3
        pass

    def initialize_params(self, X):
        self.n_samples, self.n_features = X.shape

        # randomly choose the starting centroids/means
        self.mu = X[np.random.choice(
            self.n_samples, self.n_centroids, False), :]

        # initialize the covariance matrices for each Gaussian
        self.sigma = [np.eye(self.n_features)] * self.n_centroids

        # initialize the probabilities for each Gaussian
        self.phi = [1. / self.n_centroids] * self.n_centroids

        # w matrix is initialized to all zeros
        self.w = np.zeros((self.n_samples, self.n_centroids))

    def fit(self, X):
        self.initialize_params(X)

        # log_likelihoods
        log_likelihoods = []
        means_variation = []
        # Iterate till max_iters iterations
        while len(log_likelihoods) < self.max_iters:

            # E - Step

            # Vectorized implementation of e-step equation to calculate the
            # membership for each of k -Gaussian
            for k in range(self.n_centroids):
                self.w[:, k] = self.phi[k] * \
                    self.P(self.mu[k], self.sigma[k], X)

            # Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(self.w, axis=1)))

            log_likelihoods.append(log_likelihood)

            # Normalize w matrix
            self.w = (self.w.T / np.sum(self.w, axis=1)).T

            # The number of data-points belonging to each Gaussian
            N_ks = np.sum(self.w, axis=0)

            # M Step
            # calculate the new mean and covariance for each Gaussian by
            # utilizing the new w matrix
            for k in range(self.n_centroids):

                # means
                self.mu[k] = 1. / N_ks[k] * \
                    np.sum(self.w[:, k] * X.T, axis=1).T
                means_variation.append(self.mu[k].copy())
                x_mu = np.matrix(X - self.mu[k])

                # covariances
                self.sigma[k] = np.array(
                    1 / N_ks[k] *
                    np.dot(np.multiply(x_mu.T, self.w[:, k]), x_mu))

                # and finally the phi
                self.phi[k] = 1. / self.n_samples * N_ks[k]
            # check for convergence
            if len(log_likelihoods) < 2:
                continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.convergence:
                break

        # for plotting
        return means_variation

    def predict(self, x):
        probs = np.array([phi * self.P_predict(mu, s, x) for mu, s, phi in
                          zip(self.mu, self.sigma, self.phi)])
        return probs / np.sum(probs)
