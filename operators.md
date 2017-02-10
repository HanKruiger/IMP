# Operators

In IMP, an operator is defined as follows:

## Input

A list (may be empty) of datasets.
Described to the application through the `Operator.describe_input()` method.

## Parameters

A list of parameters for the operator.
Described to the application through the `Operator.describe_parameters()` method.

## Output

A list of new datasets.
Described to the application through the `Operator.describe_output()' method.
(For now, this is always a single dataset, so this is not implemented. But for a more general approach, it should be there.)
