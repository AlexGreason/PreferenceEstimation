# PreferenceEstimation
Extends the work in my CharacterAutoencoder repository. The goal is to learn an approximation of the user's preferences regarding what looks "cool" or "interesting" or "like a real character" and then generate characters that score well by that metric. At a very early stage of development. 

Overall intended pipeline:
1. A set of random characters is generated
2. Pairs of characters are presented to the user, and the user decides which better satisfies whatever criterion they want to maximize
3. These answers are then fed into a Bradley-Terry model, which returns specific scores for each character, the difference between the scores of two characters indicating essentially how likely it is that the user, presented with that pair, would select one or the other
4. this is continued until a fairly accurate ranking is determined
5. that ranking is then used to train a neural net that attempts to approximate the user's preferences, in that it tries to reproduce the ranking
6. an evolutionary algorithm is then applied to the set of characters, with scoring performed by the neural net, to produce a new set of characters
7. repeat steps 2-7 until sufficiently good characters are generated.

The purpose of both the bradley-terry model and the neural preference estimator are to essentially provide "leverage" for the evolutionary algorithm:
with a set of 30 characters, 100-200 comparisons is enough to generate a pretty good set of rankings, which then determines the desired result for 1800 total training cases (2\*30^2), speeding training of the neural net. The neural net can then provide some value (even at very early stages of training) in guiding the evolutionary algorithm, with the neural net improving as the results of the evolutionary algorithm become more optimized, both learning in tandem. The neural net can score tens of thousands, hundreds of thousands, or even millions of characters, with the limit primarily being how much the set of characters can change before the currrent state of the preference estimator is no longer effective at scoring them. So through this pipeline, a few thousand user evaluations (a perfectly reasonable number, which should only take an hour or two of scoring) can be leveraged to provide millions of fitness-function evaluations to the evolutionary algorithm. That's the idea, anyway, it doesn't really work yet. I've completed steps 1-4, and I've got something for step 5, but the evolutionary algorithm has just been started.
