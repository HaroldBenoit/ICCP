---
layout: default
title: Why not OpenAI Gym? 
parent: Background information
nav_order: 3
permalink: /docs/background-information/openai
---

## What's OpenAI Gym ?

[OpenAI Gym](https://www.gymlibrary.ml/) is the de-facto open source toolkit for developing and comparing RL algorithms. It is widely used in the scientific community.

## Why is the Environment class not an OpenAI Gym ?

Although it is widely used in teh scientific community, its main users are computer scientists and software engineers. This has for effect that integration and usage can be somewhat tedious on certain aspects (custom envrionment installation). As this project is to be used in the context of the [ICE lab](https://www.epfl.ch/labs/ice/), its users are not trained software engineers. Thus, it was chosen to make a compromise.

The [Environment class](../../../enac-docs/docs/technical-reference/environment/environment_class) has an API almost identical to the Gym API but it doesn't require the end user to `pip install` its own environments and deal with namespace issues.

Thus, we keep the simplicity of use and:
- any Gym environment can be very easily converted to our [Environment](../../../enac-docs/docs/technical-reference/environment/environment_class) interface.

- any Gym compatible RL algorithm can be very easily converted to our [Environment](../../../enac-docs/docs/technical-reference/environment/environment_class) interface.


