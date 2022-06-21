---
layout: default
title: Performance assessment
parent: Background information
nav_order: 3
permalink: /docs/background-information/assessment
---

# Systematic Reinforcement Learning Performance Assessment


### Acknowledgments

This section was heavily inspired by the paper ["Measuring the reliability of reinforcement learning algorithms (2020)"](https://arxiv.org/pdf/1912.05663.pdf) written by Stephanie C.Y. Chan,Samuel Fishman, John Canny,Anoop Korattikara & Sergio Guadarrama.


## Introduction

Performance evaluations are critical for quantifying algorithmic advances in reinforcement learning.

As a naive approach, there’s the "scalar" / one-dimensional way of assessing performance, which might be to simply choose the RL algorithm with the highest cumulative reward.

But this approach fails to capture important details of using a RL controller in real-life which are:
- Its ease of usage and training (computational resources, time, ...)
- Its reliability and its variability (dispersion and risk)

Lack of reliability is a well-known issue for reinforcement learning (RL) algorithms. Such algorithms, especially Deep RL algorithms, tend to be highly variable in performance and considerably sensitive to a range of different factors, including implementation details, hyper-parameters, choice of environments, and even random seeds.

This variability hinders reproducible research, and can be costly or even dangerous for real-world applications. Furthermore, it impedes scientific progress in the field when practitioners cannot reliably evaluate or predict the performance of any particular algorithm, compare different algorithms, or even compare different implementations of the same algorithm.


Thus, the `Performance` module in `ICCP` aims to achieve a systematic, reproducible, and multi-dimensional performance assessment that captures the important details mentioned above by following the guidelines of the mentioned [paper](https://arxiv.org/pdf/1912.05663.pdf).

## Context 

Using three axes of variability and two measures of variability, our procedure will assess the performance of a RL algorithm at a deeper level, especially tailored for real-world usage.

Usually, this performance assessment may be ran on a RL controller that has been previously chosen for its excellent performance according to a defined [utility function](../../../enac-docs/docs/background-information/utility).


## Measures of variability

#### Dispersion 
Given a range of values, one may define its dispersion as the width of the distribution. 

One may use the standard deviation to compute it but a more robust measure was chosen: [IQR (interquartile range)](https://en.wikipedia.org/wiki/Interquartile_range)

#### Risk
Given a range of values, one may define its risk as the worst possible performance or the heaviness of the tail of the distribution.

A robust measure was chosen: [Expected shortfall](https://en.wikipedia.org/wiki/Expected_shortfall) also known as CVaR (conditional value at risk)


## Axes of variability

Our metrics target the following three axes of variability. The first two capture reliability *"during training"*, while the last captures reliability of a fixed policy i.e. *"after learning"*.

### Across Time

In the setting of evaluation during training, one desirable property for an RL algorithm is to be stable "across time" within each training run. In general, smooth monotonic improvement is preferable to noisy fluctuations around a positive trend, or unpredictable swings in performance.

This type of stability is important for several reasons. During learning, especially when deployed for real applications, it can be costly or even dangerous for an algorithm to have unpredictable levels of performance.

Even in cases where bouts of poor performance do not directly cause harm, e.g. if training in simulation, high instability implies that algorithms have to be check-pointed and evaluated more frequently in order to catch the peak performance of the algorithm, which can be expensive. 

Furthermore, while training, it can be a waste of computational resources to train an unstable algorithm that tends to forget previously learned behaviors

For example, we might like to measure how "risky" and "variable" the obtained reward by a controller is. This would allow us to quantify how much the controller has a tendency to have periods of bad performance that might affect the human users.

## Across Runs:
In the setting of evaluation during training, RL algorithms should have easily and consistently reproducible performances across multiple training runs.

Depending on the components that we allow to vary across training runs, this variability can encapsulate the algorithm’s sensitivity to a variety of factors, such as: 
- random seed 
- initialization of the optimization
- initialization of the environment 
- implementation details
- hyper-parameter settings

High variability on any of these dimensions leads to unpredictable performance, and also requires a large search in order to find a model with good performance.


## Across rollouts of a fixed policy


In the setting of evaluation after training, a natural concern is the variability in performance across multiple rollouts of a fixed policy. Generally, this axe of variability measures sensitivity to both stochasticity from the environment and stochasticity from the training procedure (the optimization).


## Conclusion

Using those three axes of variability and the two measures of variability, one can assess the performance of a RL algorithm at a deeper level, especially tailored for real-world usage.





