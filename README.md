# SHRED-ROM

This repository contains the official source code implementation of the paper *Reduced order modeling with shallow recurrent decoder networks*.

`utils` folder contains auxiliary functions to preprocess and plot data, as well as to define and train SHRED-ROM. These functions are mainly based on the [pyshred](https://github.com/Jan-Williams/pyshred) repository developed by [Jan Williams](https://github.com/Jan-Williams).
<br />
<br />
<br />


<p align="center" width="100%">
  <img width=100% src="./media/SHRED-ROM.png" >
  <br />
</p>

## Shallow Water
`SWE.ipynb` presents the Shallow Water test case where we reconstruct the high-dimensional velocity on a sphere, whose dynamics is described by the Shallow Water Equations, starting from few sensor data.

<p align="center" width="100%">
  <img width=80% src="./media/SWE.gif" >
  <br />
</p>

## GoPro physics
`GoPro.ipynb` presents GoPro physics test case where we reconstruct high-dimensional videos starting from few pixel data.

<p align="center" width="100%">
  <img width=100% src="./media/GoPro.gif" >
  <br />
</p>

## Kuramoto-Sivashinsky
`KuramotoSivashinsky.ipynb` presents the Kuramoto-Sivashinsky test case where we reconstruct the high-dimensional state, whose dynamics is described by the Kuramoto-Sivashinsky equation, starting from few sensor data while considering different viscosities and initial conditions.

<p align="center" width="100%">
  <img width=100% src="./media/KuramotoSivashinsky.gif" >
  <br />
</p>

## Fluidic pinball
`Pinball.ipynb` presents the fluidic pinball test case where we reconstruct the high-dimensional density, whose dynamics is described by the advection-diffusion partial differential equation, starting from few sensor data while considering different velocities of the three rotating cylinders.

<p align="center" width="100%">
  <img width=80% src="./media/Pinball.gif" >
  <br />
</p>

## Flow around an obstacle
`FlowAroundObstacle.ipynb` presents the flow around an obstacle test case where we reconstruct the high-dimensional velocity, whose dynamics is described by the unsteady Navier-Stokes equations, starting from few sensor data while considering different inflow conditions and obstacle geometries.

<p align="center" width="100%">
  <img width=50% src="./media/FlowAroundObstacle.gif" >
  <img width=40% src="./media/FlowAroundObstacle_paramestimation.gif" >
  <br />
</p>
