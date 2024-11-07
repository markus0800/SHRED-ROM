# SHRED-ROM

This repository contains the official source code implementation of the paper *---title---*

`utils` folder contains auxiliary functions to process and plot data, as well as to define and train the SHRED model. These functions are mainly based on the [pyshred](https://github.com/Jan-Williams/pyshred) repository developed by [Jan Williams](https://github.com/Jan-Williams)

## Kuramoto-Sivashinsky
The folder `KuramotoSivashinsky` contains the implementation of the Kuramoto-Sivashinsky test case where we reconstruct the high-dimensional Kuramoto-Sivashinsky state starting from few sensor data while considering different viscosity coefficients and noisy measurements

<p align="center" width="100%">
  <img width=80% src="./KuramotoSivashinsky/KS.png" >
  <br />
</p>

## Pinball
`Pinball.ipynb` presents the pinball test case where we reconstruct the high-dimensional density, whose dynamics is described by the Fokker-Planck equation, starting from few sensor data while considering different velocities of the three rotating cylinders

<p align="center" width="100%">
  <img width=80% src="./gifs/Pinball.gif" >
  <br />
</p>

## Flow around an obstacle
`FlowAroundObstacle.ipynb` presents the flow around an obstacle test case where we reconstruct the high-dimensional velocity, whose dynamics is described by the unsteady Navier-Stokes equation, starting from few sensors data while considering different inflow conditions and obstacle geometries

<p align="center" width="100%">
  <img width=60% src="./gifs/FlowAroundObstacle.gif" >
  <br />
</p>
