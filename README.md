# bayesian-structure-learning
Takes in a dataset of multiple variables and their instantiations. Outputs the optimal bayesian network (via networkX) using a scoring algorithm and K2 search.
My p_instantiations() function returns a list of lists where each inner list is a particular permutation of the possible instantiations of a nodes' parents. The fill_matrix() function utilizes this by creating an all-zeros counts matrix of the correct dimensions (number of possible parental instantiations x number of instantiations for that node). It then iterates through the data set only one time. At each row of the data set it takes note of the values of the node's parents (p_vals). It then finds the index (j) of p$\textunderscore$vals in the node's parental instantiations (p_instants). Since the shape of p_instants is the same as the empty counts matrix, we can use j to tell us which row in counts to increment. Within that row, I increment the node corresponding to the index of the node's data point minus one. I then add each node's individual counts matrix to a list. I create a list of matrices of the exact same shape with prior().

Using these two lists of matrices, I calculate the Bayesian score by splitting up the equation into three parts: the ri sum, the qi sum, and the n sum. Finally, I implemented the K2 search algorithm as demonstrated in the book and in class.

In the following examples, I decided to enact a maximum of four parents per node in order to limit compute time. 

Run times in seconds:
Small: 2.005723714828491 
Medium: 42.4382848739624 
Large: 9029.075842142105 (2.5 hours) 

Bayesian Scores: 
Small: -3835.6794252127916 
Medium: -42060.47640776394  
Large: -404192.31703113177 
