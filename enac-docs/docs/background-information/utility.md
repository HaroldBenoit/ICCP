---
layout: default
title: Utility
parent: Background information
nav_order: 4
permalink: /docs/background-information/utility
---


# Utility functions

In economics, utility represents the satisfaction or pleasure that consumers receive for consuming a good or service. More generally, one may define a utility function as a defined universal measure for a specific problem that lets us compare and choose between two outcomes. Generally, all things equal, one should want to maximize its utility function .

In the context of reinforcement learning, a utility function may be defined as the total obtained by the agent during the session. Or in the context of building control, it may defined as the total negative heating demand each second (remember we always to maximize the utility function, thus we would like to maximize the total negative heating demand if we would like to minimize heating). 


`Performance` uses the concept of utility function to easily compare different sessions or agents. The main reason is that it gives a lot of flexibility to the user, as he may define its own utility function (as it may heavily depend on the environment and its complexity).

In `ICCP`, we defined the utility function type signature as `Callable[[pd.DataFrame], float]` where `Callable´[[x_1, ..., x_n], y]` mathematically represents $f(x_1, ...,x_n) = y$. Thus, the function should take a pandas DataFrame and return a float.

#### Alreday defined utility functions

- Cumulative reward: `Peformance.cumulative_reward`
- Negative cumulative heating: `Performance.negative_cumulative_heating`
- Constant utility function, to be used when there is no preference between two sessions: `Performance.constant_utility`



