# Regret Analysis of Predictive Process Monitoring Techniques

This project contains ways to empirically analyze how 'regret' of a predictive process monitoring (PPM) technique develops, 
depending on properties of the technique. Regret is a measure of how much better one could have done on a specific (optimization)
problem, if a prediction would have been perfect. The project evaluates the
regret for two job sequencing problems.

Problem 1: Sum of completion times scheduling aims to sequence a given set of cases with 
deterministic (unknown) durations to minimize the sum of completion times, which also minimizes the average
case waiting times.

Problem 2: Appointment scheduling sequences cases that have random durations that may vary even for the same case. The appointment times
are set to be according to the expected case duration. In this problem we aim to minimize both the waiting times and the idle time of the resources.

In both these problems PPM can be used to predict the durations of the tasks that need to be scheduled. The regret then is
the extent to which the solution of the problem can be improved, in terms of its sum of completion times (problem 1) or waiting times and idle times (problem 2),
if perfect predictions existed.

In this project the goodness of the prediction is expressed in terms of the Mean Squared Error (MSE) of the PPM technique.  