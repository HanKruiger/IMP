# IMP

## Run

First, create a virtual environment and (locally) install the required packages:

```bash
python3 -m venv imp_venv
source ./imp_venv/bin/activate
pip3 install -r requirements.txt
```

Then run with:
```bash
./imp.py
```

Interactive Multiscale Projections

## Roadmap

* Test point scattering in Python + Qt + OpenGL of about 10000 points.
* Further work out the application design (usage-wise, then implementation-wise).
* Moment of reflection.
* Develop interaction aspects (D1)
* Develop explanatory aspects (D2)
* Develop parallelization aspects (D3, at VMware, if it fits with their application)

## Notation

* `X`: Set of `N` `m`-dimensional observations.
* `N`: Number of input observations.
* `m`: Dimensionality of input data.
* `N_r`: Number of representatives shown to the user. The algorithm aims at showing at most this number of points on the screen at all times.

## Algorithm/application:

Input:
* `X`

Parameters:
* `N_r`

Workflow:

* For the first presentation shown to the user, `X` is clustered in a hierarchical top-down fashion, until there are `N_r` representatives. The clustering algorithm should:
  * be not too expensive.
  * be refinable. That is, once some intermediate-level clustering is made, it should be possible to zoom in to parts of the projection that can be clustered further.
  * cluster uniformly over the high-dimensional manifold.
* The `N_r` representatives from the previous step are projected to 2D (with some suitable projection technique) and are shown to the user.  The projection technique should:
  * be not too expensive.
  * preserve a relevant data pattern. E.g.:
    * Pairwise distances
    * Neighbourhoods
* The user is able to select a 2D region where she wants to zoom in to. When this happens, there are two possibilities:
  *  If one or more of the current representatives reside in the 2D region, a new set of representatives is defined by recursively ‘inflating’ the representatives residing in the 2D region until there are `N_r` representatives. This ‘inflation’ is the actual on-the-fly hierarchical clustering. After the inflation, the projection of the new representatives is computed and scaled s.t. the projection fills the 2D screen.
    * To choose the representatives that are ‘inflated’, some strategy is used. E.g.:
      * Higher-level representatives first.
      * Representatives representing more data-points first.
      * Representatives representing data-points that are very uncluster-like (by some criterion) first.
    * For better usability, the visualization of the projection should somehow be interpolated from the current to the next. That way it is clear where the next points in the projection ‘come from’ (continuity). This could be problematic.
    * The previous representatives (in the 2D region) may act as control points for the new set of representatives. (E.g., for LAMP, or other projection with control points.)
  * If no representatives reside in the 2D region, a set of representatives is defined by ‘transforming’ the 2D region to an `m`D-region, and querying for points there.
    * How to make this transformation?
      * You have a bounding box in 2D. This allows to compute the 2D centroid of that bounding box. I don’t see a straightforward way to transform this to an `m`D centroid.

## Languages and libraries

|                                             | C++                           | Python             | JavaScript               |
|:--------------------------------------------|:------------------------------|:-------------------|:-------------------------|
| OpenGL                                      | Yes                           | Yes                | WebGL (OpenGLES 2.0)     |
| CUDA                                        | Yes                           | Yes                | No, but WebCL            |
| GLSL shaders                                | Yes                           | Yes                | Yes, but limited         |
| GUI                                         | Qt                            | Qt                 | HTML+CSS                 |
| Linear Algebra & Machine Learning libraries | Many independent libraries    | SciKit             | Probably no decent ones  |
| Expected performance                        | ++                            | +                  | -                        |
| Shareability of application                 | +-                            | +                  | ++                       |
| Other                                       | Programming is more type-work | Familiar with this | Work with this at VMware |

## HSNE (for comparison)

* `L_i`: Set of landmarks for level `i`.
* `T_i`: Finite Markov Chain of transitions in level `i`.
* `pi_i` Equilibrium distribution of `T_i`.

```
Start with L_1, which is the set of all data points.

Build T_1 from the kNN-graph of L_1.

s = 1

while (too many details in projection) do:
  Sample the equilibrium distribution pi_s of T_s, by using random walks.
  
  From pi_s, select high values to be in L_{s+1}. (Low values are discarded as outliers.)
  
  Compute area of influence for each landmark in L_{s+1} on landmarks in L_s by using random walks.
  
  Build T_{s+1} by considering overlaps between areas of influence.

  s += 1
```

Notable differences between HSNE and IMP:

* HSNE has a bottom-up approach.
* Landmark selection in HSNE uses convoluted heuristics.
