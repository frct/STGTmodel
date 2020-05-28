# STGTmodel
 Models and code from Sign-tracking vs. goal-tracking Cinotti et al.  2019

The project is split into four different folders which correspond to different variations of the model:

      - standard model: the original model by Lesaint et al. (2014) translated into python (original code was in C++) with a few minor adjustments. This is the version used throughout most of the paper
      
      - sequential actions model is a first attempt to explain why in the short ITI condition, animals switch from goal-tracking to sign-tracking. To study this, I allowed agents in state 1, when the lever appears, to take two different actions before changing state. The result was that switching from sign-tracking to goal tracking is possible but not the opposite, as it is disadvantageous from a MB point of view.
      
      - extra-states model, similar in aim to sequential actions model, this is the model used in figure 6 of the paper which also fails to account for the experimental result.
      
      - decaying MB is an idea I played with for a bit, concerning the possibility that goal-tracking is a transient unstable behaviour which is eventually replaced by sign-tracking. Experimental support for this hypothesis is disputed, but we wanted to theoretically pursue it as an illustration of how a model-based system can be replaced by a model free one.
