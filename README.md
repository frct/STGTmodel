# STGTmodel
 Models and code for collaboration project with S. Groman and D. Calu's teams
 
 The aim of this project is to relate the sign-tracking vs goal-tracking behaviour of rats in a Pavlovian Conditioned Approach task with their model-based vs model-free tendencies in a two-stage decision task specially adapted to rats. The STGT model proposes to explain behaviour in the PCA task as a balance between model-based and model-free learning processes, hence its relevance to the project. More precisely, in this part of the project, we optimise the STGT model on each rat to extract its own individual omega parameter which reflects its tendency to favour model-free behaviour.
The project is made up of three scripts:
- TaskVariables which defines the different possible states, actions, features and state-action transitions, etc.
- ModelDefinition with the model-free and model-based learning modules
- ModelOptimisation which given the experimental data, finds an optimal value of omega to minimize the negative log-likelihood
(-ModelSimulation which given an optimised set of parameters will generate new behaviour for comparison with the experimental results)
