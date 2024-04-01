# Permutation Flowshop Solution Implementation

Consider a classic problem in the literature, the Permutation Flowshop (PFSP) in which a set of J tasks consists of a
set of tasks (1, 2, ..., j) to be processed on a set M of machines (1, 2, ..., m).
In the PFSP, all tasks are processed
serially on multiple machines in the same order.
Moreover, each job can be processed on a single machine at a time and
each machine can process only one job at a time, respectively.
In addition, all operations cannot be interrupted, and
setup times are included in the processing times and are independent of sequencing decisions.
The scheduling problem is
to find a task sequence that optimizes a certain performance criterion, in this case, the total time to complete all
tasks (makespan).

We are given the following standard problem in the literature Ta001 (Taillard, 1993), which is a 20-task, 5-machine
problem.

|    | J1 | J2 | J3 | J4 | J5 | J6 | J7 | J8 | J9 | J10 | J11 | J12 | J13 | J14 | J15 | J16 | J17 | J18 | J19 | J20 |
|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| M1 | 54 | 83 | 15 | 71 | 77 | 36 | 53 | 38 | 27 | 87  | 76  | 91  | 14  | 29  | 12  | 77  | 32  | 87  | 68  | 94  |
| M2 | 79 | 3  | 11 | 99 | 56 | 70 | 99 | 60 | 5  | 56  | 3   | 61  | 73  | 75  | 47  | 14  | 21  | 86  | 5   | 77  |
| M3 | 16 | 89 | 49 | 15 | 89 | 45 | 60 | 23 | 57 | 64  | 7   | 1   | 63  | 41  | 63  | 47  | 26  | 75  | 77  | 40  |
| M4 | 66 | 58 | 31 | 68 | 78 | 91 | 13 | 59 | 49 | 85  | 85  | 9   | 39  | 41  | 56  | 40  | 54  | 77  | 51  | 31  |
| M5 | 58 | 56 | 20 | 85 | 53 | 35 | 53 | 41 | 69 | 13  | 86  | 72  | 8   | 49  | 47  | 87  | 58  | 18  | 68  | 28  |

According to this problem, the processing time of job 3 (J3) on machine 5
(M5) is for example 20 units of time.
Questions:

- Calculate the number of possible solutions to the problem
- Explain which type of algorithm best suits its solution (exact,
  heuristics, metaheuristics).
- Choose an algorithm to solve and develop it in the language
  programming language of your choice.
- List the optimal solution you found and the makespan (total time
  completion time) of this solution*.

* given that BKS = 1278 where BKS -> Best Known Solution

# Solution

The solution we got is **1278** in around **1** second, which is the same as the Best Known Solution (BKS) for this problem. The algorithm used to
solve this problem is a hybrid algorithm that combines the Genetic Algorithm (GA) with the Tabu Search (TS) algorithm.

The code is implemented in Python and can be found in the following
link: [Permutation Flowshop Solution Implementation](final.py)

The analytical explanation of the code can be found in the following
link: [Permutation Flowshop Solution Implementation Explanation](pfsp.ipynb)