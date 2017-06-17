#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16  01:49:49 2017

@author: mortza
"""
import numpy as np
from . import config


class EM():
    """
    expectation maximization class for GMM
    """

    def __init__(self, k, convergence=1e-5, prior_prob='auto'):
        """initialize a EM object

        Parameters
        ----------
        k : int
            number of clusters (centroids)
        convergence : float, optional
            convergence threshold
        prior_prob : str or array-like, optional
            represent prior probabilities of each distribution ( p(z(i)=j)'s )
            if set to 'auto' it will assign equal probability for all
            zi and xi's
            if passed array must be in shape of (n_samples, k)
        """
        self.phi = np.zeros(k)
        self.convergence = convergence
        self.n_centroids = k
        if prior_prob == 'auto':
            self.phi = None
        else:
            # TODO : check for __iter__ method
            try:
                self.phi = np.ndarray(prior_prob)
            except Exception as ex:
                print(f'''{ex}, prior probabilities set to 'auto' ''')
                self.phi = None

    def initialize_(self, X):
        """initialize EM parameters

        Parameters
        ----------
        X : numpy.ndarray
        """
        self.n_samples, self.n_features = X.shape
        # initialize mu
        self.mu = X[np.random.choice(
            self.n_samples, self.n_centroids, False), :]

        # initialize sigmas
        self.sigma = np.array([np.eye(self.n_features)] * self.n_centroids)

        if self.phi is None:
            # initialize weights
            self.phi = np.ones(
                (self.n_samples, self.n_centroids)) / self.n_centroids

        # initialize weights matrix
        self.w = np.zeros((self.n_samples, self.n_centroids))

        # responsibility matrix is initialized to all zeros
        self.R = np.zeros((self.n_samples, self.n_centroids))

        if config.DEBUG:
            print(f'n_samples {self.n_samples}')
            print(f'n_features {self.n_features}')
            print(f'mu shape {self.mu.shape}')
            print(f'sigma {self.sigma.shape}')
            print(f'z {self.z.shape}')
            print(f'w {self.w.shape}')
            print(f'R {self.R.shape}')

    def P(self, mu, sigma, xi):
        """p(X=xi|zi=j)

        Parameters
        ----------
        mu : float
            mean of cluster
        sigma : array
            array-like object with shape (self.n_features, self.n_features)
        xi : array
            array with shape (1, self.n_features)

        No Longer Returned
        ------------------
        np.ndarray
            probabilities array with shape (self.n_samples, 1)
        """
        t1 = (np.linalg.det(sigma)**0.5) * ((2 * np.pi)**(self.n_features / 2))
        t1 = 1 / t1
        t2 = np.dot(xi - mu, np.linalg.inv(sigma))
        t2 = np.exp(-0.5 * np.dot(t2, xi - mu))
        return t1 * t2

    def fit(self, X):
        """fit EM object to model

        Parameters
        ----------
        X : numpy.ndarray
            training examples with shape (n, m) where n denote number of
            training examples and m is number of features
        """
        self.initialize_(X)

        while True:
            # E step
            # calculate w
            for i in range(self.n_samples):
                for j in range(self.n_centroids):
                    # p(X=xi|zi=j)
                    # latex equation:
                    # p(x^{(i)}|z^{(i)}=j;\mu, \Sigma)p(z^{(i)}=j;\phi)
                    self.w[i, j] = self.P(self.mu[j], self.sigma[
                                          j], X[i]) * self.phi[i, j]
                # latex equation:
                # \sum_{l=1}^{k}p(x^{(i)}|z^{(i)}=l;\mu,\Sigma)p(z^{(i)}=l;\phi)
                self.w[i, j] = self.w[i, j] / np.sum(self.w[i, :])

            # M step
            # tune parameters
