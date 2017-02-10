## Operators

In IMP, an operator is defined as follows:

### Input

A list (may be empty) of datasets that are used by the operator as inputs.
Described to the application through the `Operator.describe_input()` method.

### Parameters

A list (may be empty) of parameters for the operator.
Described to the application through the `Operator.describe_parameters()` method.

### Output

A list of datasets that are the result of this operator.
Described to the application through the `Operator.describe_output()` method.
(For now, this is always a single dataset, so this is not implemented. But for a more general approach, it should be there.)

## How to use an operator

* Initialize an `Operator` object: `operation = Operator()`
* Set the input: `operation.set_input(dataset1, dataset2)`
* Set the parameters: `operation.set_parameters(p1, p2, p3)`
* The operation can be started by calling the `operation.run()` method. It will execute in a seperate thread.

## Semantics

* An __operator__ is a class.
* An __operation__ is an object. This is the operator, bound to some specified input/parameters/output.
