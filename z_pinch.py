import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import time


"""
Very simple 3D MHD code for simulating shear flow stabilized Z-pinch

This code was modified from the 2D MHD code by Philip Mocz (2023), @PMocz
Reference:
    git@github.com:loliverhennigh/ConstrainedTransportZPinch.git
"""


def write_vtk(
    rho,
    bx,
    by,
    bz,
    filename="state.vtk",
    origin=(0.0, 0.0, 0.0),
    dx=1.0,
):
    """
    Write density and magnetic field to a VTK file.
    """

    # Get numpy arrays
    rho = np.array(jax.device_get(rho))
    bx = np.array(jax.device_get(bx))
    by = np.array(jax.device_get(by))
    bz = np.array(jax.device_get(bz))

    # Get the dimensions of the data
    nx, ny, nz = rho.shape

    # Create a UniformGrid
    grid = pv.ImageData()

    # Set the grid dimensions (note the +1 for cell-centered data)
    grid.dimensions = np.array(rho.shape) + 1  # Dimensions are points, so add 1

    # Set the spacing between points
    grid.spacing = (dx, dx, dx)

    # Set the origin of the grid
    grid.origin = origin

    # Flatten the density data in Fortran order (column-major)
    grid.cell_data["rho"] = rho.flatten(order="F")

    # Stack the magnetic field components and flatten
    B = np.stack((bx, by, bz), axis=-1)  # Shape: (nx, ny, nz, 3)
    B_flat = B.reshape(-1, 3, order="F")

    # Assign the magnetic field as a vector field
    grid.cell_data["B"] = B_flat

    # Save the grid to a VTK file
    grid.save(filename)


@jax.jit
def get_curl(Ax, Ay, Az, dx):
    """
    Calculate the discrete curl in 3D
    Ax, Ay, Az are matrices of nodal x, y, z-components of magnetic potential
    dx       is the cell size
    bx, by, bz are matrices of cell face x, y, z-components magnetic-field
    """

    # Magnetic field components
    bx = (Az - jnp.roll(Az, 1, axis=1)) / dx - (
        Ay - jnp.roll(Ay, 1, axis=2)
    ) / dx  # left/down roll
    by = (Ax - jnp.roll(Ax, 1, axis=2)) / dx - (Az - jnp.roll(Az, 1, axis=0)) / dx
    bz = (Ay - jnp.roll(Ay, 1, axis=0)) / dx - (Ax - jnp.roll(Ax, 1, axis=1)) / dx

    return bx, by, bz


@jax.jit
def get_div(bx, by, bz, dx):
    """
    Calculate the discrete curl of each cell
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    bz       is matrix of cell face z-component magnetic-field
    dx       is the cell size
    """

    divB = (
        bx
        - jnp.roll(bx, 1, axis=0)  # left/down roll
        + by
        - jnp.roll(by, 1, axis=1)
        + bz
        - jnp.roll(bz, 1, axis=2)
    ) / dx

    return divB


@jax.jit
def get_B_avg(bx, by, bz):
    """
    Calculate the volume-averaged magnetic field
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    bz       is matrix of cell face z-component magnetic-field
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    Bz       is matrix of cell Bz
    """

    Bx = 0.5 * (bx + jnp.roll(bx, 1, axis=0))
    By = 0.5 * (by + jnp.roll(by, 1, axis=1))
    Bz = 0.5 * (bz + jnp.roll(bz, 1, axis=2))

    return Bx, By, Bz


@jax.jit
def get_conserved(rho, vx, vy, vz, P, Bx, By, Bz, gamma, vol):
    """
    Calculate the conserved variable from the primitive
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    vz       is matrix of cell z-velocity
    P        is matrix of cell Total pressures
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    Bz       is matrix of cell Bz
    gamma    is ideal gas gamma
    vol      is cell volume
    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Energy   is matrix of energy in cells
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Momz = rho * vz * vol
    Energy = (
        (P - 0.5 * (Bx**2 + By**2 + Bz**2)) / (gamma - 1)
        + 0.5 * rho * (vx**2 + vy**2 + vz**2)
        + 0.5 * (Bx**2 + By**2 + Bz**2)
    ) * vol

    return Mass, Momx, Momy, Momz, Energy


@jax.jit
def get_primitive(Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol):
    """
    Calculate the primitive variable from the conservative
    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Momz     is matrix of z-momentum in cells
    Energy   is matrix of energy in cells
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    Bz       is matrix of cell Bz
    gamma    is ideal gas gamma
    vol      is cell volume
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    vz       is matrix of cell z-velocity
    P        is matrix of cell Total pressures
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    vz = Momz / rho / vol
    P = (
        Energy / vol
        - 0.5 * rho * (vx**2 + vy**2 + vz**2)
        - 0.5 * (Bx**2 + By**2 + Bz**2)
    ) * (gamma - 1) + 0.5 * (Bx**2 + By**2 + Bz**2)

    return rho, vx, vy, vz, P


@jax.jit
def get_gradient(f, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    f_dz     is a matrix of derivative of f in the z-direction
    """

    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (
        2 * dx
    )  # (right - left) / 2dx
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dx)
    f_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2 * dx)

    return f_dx, f_dy, f_dz


@jax.jit
def slope_limiter(f, dx, f_dx, f_dy, f_dz):
    """
    Apply slope limiter to slopes (minmod)
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    f_dz     is a matrix of derivative of f in the z-direction
    """

    eps = 1.0e-12

    # Keep a copy of the original slopes
    orig_f_dx = f_dx
    orig_f_dy = f_dy
    orig_f_dz = f_dz

    # Function to adjust the denominator safely
    def adjust_denominator(denom):
        denom_safe = jnp.where(
            denom > 0, denom + eps, jnp.where(denom < 0, denom - eps, eps)
        )
        return denom_safe

    # For x-direction
    denom = adjust_denominator(orig_f_dx)
    num = (f - jnp.roll(f, 1, axis=0)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    ratio = num / denom  # Use the same adjusted denominator
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    # For y-direction
    denom = adjust_denominator(orig_f_dy)
    num = (f - jnp.roll(f, 1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    # For z-direction
    denom = adjust_denominator(orig_f_dz)
    num = (f - jnp.roll(f, 1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    return f_dx, f_dy, f_dz


@jax.jit
def extrapolate_to_face(f, f_dx, f_dy, f_dz, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    f_dz     is a matrix of the field z-derivatives
    dx       is the cell size
    f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis
    f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis
    f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis
    f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis
    f_ZL     is a matrix of spatial-extrapolated values on `left' face along z-axis
    f_ZR     is a matrix of spatial-extrapolated values on `right' face along z-axis
    """
    f_XL = f - f_dx * dx / 2
    f_XL = jnp.roll(f_XL, -1, axis=0)  # right/up roll
    f_XR = f + f_dx * dx / 2

    f_YL = f - f_dy * dx / 2
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2

    f_ZL = f - f_dz * dx / 2
    f_ZL = jnp.roll(f_ZL, -1, axis=2)
    f_ZR = f + f_dz * dx / 2

    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    """
    Apply fluxes to conserved variables
    F        is a matrix of the conserved variable field
    flux_F_X is a matrix of the x-dir fluxes
    flux_F_Y is a matrix of the y-dir fluxes
    flux_F_Z is a matrix of the z-dir fluxes
    dx       is the cell size
    dt       is the timestep
    """

    # update solution
    F += -dt * dx**2 * flux_F_X
    F += dt * dx**2 * jnp.roll(flux_F_X, 1, axis=0)  # left/down roll
    F += -dt * dx**2 * flux_F_Y
    F += dt * dx**2 * jnp.roll(flux_F_Y, 1, axis=1)
    F += -dt * dx**2 * flux_F_Z
    F += dt * dx**2 * jnp.roll(flux_F_Z, 1, axis=2)

    return F


@jax.jit
def constrained_transport(
    bx, by, bz, flux_By_X, flux_Bx_Y, flux_Bz_X, flux_By_Z, flux_Bx_Z, flux_Bz_Y, dx, dt
):
    """
    Apply fluxes to face-centered magnetic fields in a constrained transport manner in 3D
    bx, by, bz        are matrices of cell face magnetic-field components
    flux_* are matrices of the dir fluxes of B components
    dx                is the cell size
    dt                is the timestep
    """

    # compute electric fields at nodes
    Ex = 0.25 * (
        flux_Bz_Y
        + jnp.roll(flux_Bz_Y, -1, axis=2)  # right/up roll
        - flux_By_Z
        - jnp.roll(flux_By_Z, -1, axis=1)
    )
    Ey = 0.25 * (
        flux_Bx_Z
        + jnp.roll(flux_Bx_Z, -1, axis=0)
        - flux_Bz_X
        - jnp.roll(flux_Bz_X, -1, axis=2)
    )
    Ez = 0.25 * (
        flux_By_X
        + jnp.roll(flux_By_X, -1, axis=1)
        - flux_Bx_Y
        - jnp.roll(flux_Bx_Y, -1, axis=0)
    )

    # compute db components
    dbx, dby, dbz = get_curl(Ex, Ey, Ez, dx)

    # update magnetic fields
    bx += dt * dbx
    by += dt * dby
    bz += dt * dbz

    return bx, by, bz


@jax.jit
def get_flux(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    vz_L,
    vz_R,
    P_L,
    P_R,
    Bx_L,
    Bx_R,
    By_L,
    By_R,
    Bz_L,
    Bz_R,
    gamma,
):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    vz_L         is a matrix of left-state  z-velocity
    vz_R         is a matrix of right-state z-velocity
    P_L          is a matrix of left-state  Total pressure
    P_R          is a matrix of right-state Total pressure
    Bx_L         is a matrix of left-state  x-magnetic-field
    Bx_R         is a matrix of right-state x-magnetic-field
    By_L         is a matrix of left-state  y-magnetic-field
    By_R         is a matrix of right-state y-magnetic-field
    Bz_L         is a matrix of left-state  z-magnetic-field
    Bz_R         is a matrix of right-state z-magnetic-field
    gamma        is the ideal gas gamma
    flux_Mass    is the matrix of mass fluxes
    flux_Momx    is the matrix of x-momentum fluxes
    flux_Momy    is the matrix of y-momentum fluxes
    flux_Momz    is the matrix of z-momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_L = (
        (P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)) / (gamma - 1)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)) / (gamma - 1)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)
        + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)
    )

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)
    Bz_star = 0.5 * (Bz_L + Bz_R)
    P_star = (gamma - 1) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2 + momz_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star
    flux_Momz = momx_star * momz_star / rho_star - Bx_star * Bz_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star + Bz_star * momz_star
    ) / rho_star
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star
    flux_Bz = (Bz_star * momx_star - Bx_star * momz_star) / rho_star

    # Find wave speeds
    c0_L = jnp.sqrt(gamma * (P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)) / rho_L)
    c0_R = jnp.sqrt(gamma * (P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)) / rho_R)
    ca_L = jnp.sqrt((Bx_L**2 + By_L**2 + Bz_L**2) / rho_L)
    ca_R = jnp.sqrt((Bx_R**2 + By_R**2 + Bz_R**2) / rho_R)
    cf_L = jnp.sqrt(
        0.5 * (c0_L**2 + ca_L**2) + 0.5 * jnp.sqrt((c0_L**2 + ca_L**2) ** 2)
    )
    cf_R = jnp.sqrt(
        0.5 * (c0_R**2 + ca_R**2) + 0.5 * jnp.sqrt((c0_R**2 + ca_R**2) ** 2)
    )
    C_L = cf_L + jnp.abs(vx_L)
    C_R = cf_R + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Momz -= C * 0.5 * (rho_L * vz_L - rho_R * vz_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    flux_By -= C * 0.5 * (By_L - By_R)
    flux_Bz -= C * 0.5 * (Bz_L - Bz_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_By, flux_Bz


@jax.jit
def update(bx, by, bz, Mass, Momx, Momy, Momz, Energy, vol, dx, t, gamma, courant_fac):
    # get B cell-centered values
    Bx, By, Bz = get_B_avg(bx, by, bz)

    # get Primitive variables
    rho, vx, vy, vz, P = get_primitive(
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol
    )

    # get time step (CFL) = dx / max signal speed
    c0 = jnp.sqrt(gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) / rho + 1e-16)
    ca = jnp.sqrt((Bx**2 + By**2 + Bz**2) / rho + 1e-16)
    cf = jnp.sqrt(
        0.5 * (c0**2 + ca**2) + 0.5 * jnp.sqrt((c0**2 + ca**2) ** 2 + 1e-16)
    )
    dt = courant_fac * jnp.min(dx / (cf + jnp.sqrt(vx**2 + vy**2 + vz**2) + 1e-8))

    # calculate gradients
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx, vx_dy, vx_dz = get_gradient(vx, dx)
    vy_dx, vy_dy, vy_dz = get_gradient(vy, dx)
    vz_dx, vz_dy, vz_dz = get_gradient(vz, dx)
    P_dx, P_dy, P_dz = get_gradient(P, dx)
    Bx_dx, Bx_dy, Bx_dz = get_gradient(Bx, dx)
    By_dx, By_dy, By_dz = get_gradient(By, dx)
    Bz_dx, Bz_dy, Bz_dz = get_gradient(Bz, dx)

    # slope limit gradients
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx, vx_dy, vx_dz = slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
    vy_dx, vy_dy, vy_dz = slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
    vz_dx, vz_dy, vz_dz = slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)
    P_dx, P_dy, P_dz = slope_limiter(P, dx, P_dx, P_dy, P_dz)
    Bx_dx, Bx_dy, Bx_dz = slope_limiter(Bx, dx, Bx_dx, Bx_dy, Bx_dz)
    By_dx, By_dy, By_dz = slope_limiter(By, dx, By_dx, By_dy, By_dz)
    Bz_dx, Bz_dy, Bz_dz = slope_limiter(Bz, dx, Bz_dx, Bz_dy, Bz_dz)

    # extrapolate rho half-step in time
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx
        + rho * vx_dx
        + vy * rho_dy
        + rho * vy_dy
        + vz * rho_dz
        + rho * vz_dz
    )

    # extrapolate velocity half-step in time
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx
        + vy * vx_dy
        + vz * vx_dz
        + (1 / rho) * P_dx
        - (Bx / rho) * (2 * Bx_dx + By_dy + Bz_dz)
        - (By / rho) * Bx_dy
        - (Bz / rho) * Bx_dz
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx
        + vy * vy_dy
        + vz * vy_dz
        + (1 / rho) * P_dy
        - (Bx / rho) * By_dx
        - (By / rho) * (Bx_dx + 2 * By_dy + Bz_dz)
        - (Bz / rho) * By_dz
    )
    vz_prime = vz - 0.5 * dt * (
        vx * vz_dx
        + vy * vz_dy
        + vz * vz_dz
        + (1 / rho) * P_dz
        - (Bx / rho) * Bz_dx
        - (By / rho) * Bz_dy
        - (Bz / rho) * (Bx_dx + By_dy + 2 * Bz_dz)
    )

    # extrapolate pressure half-step in time
    vx_dx_term = (
        gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + By**2 + Bz**2
    ) * vx_dx
    vy_dx_term = -Bx * By * vy_dx
    vz_dx_term = -Bx * Bz * vz_dx
    P_dx_term = vx * P_dx
    Bx_dx_term = (gamma - 2) * (Bx * vx + By * vy + Bz * vz) * Bx_dx
    vx_dy_term = -By * Bx * vx_dy
    vy_dy_term = (
        gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + Bx**2 + Bz**2
    ) * vy_dy
    vz_dy_term = -By * Bz * vz_dy
    P_dy_term = vy * P_dy
    Bx_dy_term = (gamma - 2) * (Bx * vx + By * vy + Bz * vz) * By_dy
    vx_dz_term = -Bz * Bx * vx_dz
    vy_dz_term = -Bz * By * vy_dz
    vz_dz_term = (
        gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + Bx**2 + By**2
    ) * vz_dz
    P_dz_term = vz * P_dz
    Bx_dz_term = (gamma - 2) * (Bx * vx + By * vy + Bz * vz) * Bz_dz
    P_prime = P - 0.5 * dt * (
        vx_dx_term
        + vy_dx_term
        + vz_dx_term
        + P_dx_term
        + Bx_dx_term
        + vx_dy_term
        + vy_dy_term
        + vz_dy_term
        + P_dy_term
        + Bx_dy_term
        + vx_dz_term
        + vy_dz_term
        + vz_dz_term
        + P_dz_term
        + Bx_dz_term
    )

    # extrapolate magnetic field half-step in time
    Bx_prime = Bx - 0.5 * dt * (
        Bx * vy_dy
        + Bx * vz_dz
        - By * vx_dy
        - Bz * vx_dz
        - vx * By_dy
        - vx * Bz_dz
        + vy * Bx_dy
        + vz * Bx_dz
    )
    By_prime = By - 0.5 * dt * (
        -Bx * vy_dx
        + By * vx_dx
        + By * vz_dz
        - Bz * vy_dz
        + vx * By_dx
        - vy * Bx_dx
        - vy * Bz_dz
        + vz * By_dz
    )
    Bz_prime = Bz - 0.5 * dt * (
        -Bx * vz_dx
        - By * vz_dy
        + Bz * vx_dx
        + Bz * vy_dy
        + vx * Bz_dx
        + vy * Bz_dy
        - vz * Bx_dx
        - vz * By_dy
    )

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrapolate_to_face(
        rho_prime, rho_dx, rho_dy, rho_dz, dx
    )
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrapolate_to_face(
        vx_prime, vx_dx, vx_dy, vx_dz, dx
    )
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrapolate_to_face(
        vy_prime, vy_dx, vy_dy, vy_dz, dx
    )
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrapolate_to_face(
        vz_prime, vz_dx, vz_dy, vz_dz, dx
    )
    P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = extrapolate_to_face(
        P_prime, P_dx, P_dy, P_dz, dx
    )
    Bx_XL, Bx_XR, Bx_YL, Bx_YR, Bx_ZL, Bx_ZR = extrapolate_to_face(
        Bx_prime, Bx_dx, Bx_dy, Bx_dz, dx
    )
    By_XL, By_XR, By_YL, By_YR, By_ZL, By_ZR = extrapolate_to_face(
        By_prime, By_dx, By_dy, By_dz, dx
    )
    Bz_XL, Bz_XR, Bz_YL, Bz_YR, Bz_ZL, Bz_ZR = extrapolate_to_face(
        Bz_prime, Bz_dx, Bz_dy, Bz_dz, dx
    )

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    (
        flux_Mass_X,
        flux_Momx_X,
        flux_Momy_X,
        flux_Momz_X,
        flux_Energy_X,
        flux_By_X,
        flux_Bz_X,
    ) = get_flux(
        rho_XL,
        rho_XR,
        vx_XL,
        vx_XR,
        vy_XL,
        vy_XR,
        vz_XL,
        vz_XR,
        P_XL,
        P_XR,
        Bx_XL,
        Bx_XR,
        By_XL,
        By_XR,
        Bz_XL,
        Bz_XR,
        gamma,
    )
    (
        flux_Mass_Y,
        flux_Momy_Y,
        flux_Momx_Y,
        flux_Momz_Y,
        flux_Energy_Y,
        flux_Bx_Y,
        flux_Bz_Y,
    ) = get_flux(
        rho_YL,
        rho_YR,
        vy_YL,
        vy_YR,
        vx_YL,
        vx_YR,
        vz_YL,
        vz_YR,
        P_YL,
        P_YR,
        By_YL,
        By_YR,
        Bx_YL,
        Bx_YR,
        Bz_YL,
        Bz_YR,
        gamma,
    )
    (
        flux_Mass_Z,
        flux_Momz_Z,
        flux_Momx_Z,
        flux_Momy_Z,
        flux_Energy_Z,
        flux_Bx_Z,
        flux_By_Z,
    ) = get_flux(
        rho_ZL,
        rho_ZR,
        vz_ZL,
        vz_ZR,
        vx_ZL,
        vx_ZR,
        vy_ZL,
        vy_ZR,
        P_ZL,
        P_ZR,
        Bz_ZL,
        Bz_ZR,
        Bx_ZL,
        Bx_ZR,
        By_ZL,
        By_ZR,
        gamma,
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, flux_Energy_Z, dx, dt)
    bx, by, bz = constrained_transport(
        bx,
        by,
        bz,
        flux_By_X,
        flux_Bx_Y,
        flux_Bz_X,
        flux_By_Z,
        flux_Bx_Z,
        flux_Bz_Y,
        dx,
        dt,
    )

    # check div B
    divB = get_div(bx, by, bz, dx)

    return bx, by, bz, Mass, Momx, Momy, Momz, Energy, divB, rho, dt


@jax.jit
def compute_divergence(P_dx, P_dy, P_dz, dx):
    P_dx_dx, _, _ = get_gradient(P_dx, dx)
    _, P_dy_dy, _ = get_gradient(P_dy, dx)
    _, _, P_dz_dz = get_gradient(P_dz, dx)
    div_P_grad = P_dx_dx + P_dy_dy + P_dz_dz
    return div_P_grad


@jax.jit
def solve_poisson_fft(div_P_grad, dx):
    # Compute the Fourier transform of div_P_grad
    div_P_grad_hat = jnp.fft.fftn(div_P_grad)

    # Get the grid sizes
    nx, ny, nz = div_P_grad.shape

    # Create the wave number grids
    kx = jnp.fft.fftfreq(nx) * 2 * jnp.pi / dx
    ky = jnp.fft.fftfreq(ny) * 2 * jnp.pi / dx
    kz = jnp.fft.fftfreq(nz) * 2 * jnp.pi / dx
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # Compute k_squared
    k_squared = kx**2 + ky**2 + kz**2
    # Avoid division by zero at the zero frequency
    k_squared = jnp.where(k_squared == 0, 1e-10, k_squared)

    # Solve for P in Fourier space
    P_hat = -div_P_grad_hat / k_squared

    # Inverse Fourier transform to get P
    P = jnp.real(jnp.fft.ifftn(P_hat))
    return P


@jax.jit
def initialize_p(Bx, By, Bz, dx, P_0):
    # Get derivatives of B
    Bx_dx, Bx_dy, Bx_dz = get_gradient(Bx, dx)
    By_dx, By_dy, By_dz = get_gradient(By, dx)
    Bz_dx, Bz_dy, Bz_dz = get_gradient(Bz, dx)

    # Get derivatives of P
    P_dx = Bx * (2 * Bx_dx + By_dy + Bz_dz) + By * Bx_dy + Bz * Bx_dz
    P_dy = Bx * By_dx + By * (Bx_dx + 2 * By_dy + Bz_dz) + Bz * By_dz
    P_dz = Bx * Bz_dx + By * Bz_dy + Bz * (Bx_dx + By_dy + 2 * Bz_dz)

    # Compute divergence of P_grad
    div_P_grad = compute_divergence(P_dx, P_dy, P_dz, dx)

    # Solve Poisson's equation to get P
    P = solve_poisson_fft(div_P_grad, dx)

    return P + 1.0


def main():
    """Finite Volume simulation"""

    # Simulation parameters
    N = 8  # Cells per unit length
    gamma = 5.0 / 3.0  # ideal gas gamma
    courant_fac = 0.4
    solve_time = 10.0
    save_animation_path = "stabalized_z_pinch"

    # Geometry parameters
    dx = 1.0 / N
    r_0 = 1.0
    x_length = 16 * r_0
    z_length = 16 * r_0
    B_0 = 1.0
    P_0 = 1.0
    rho_0 = 1.0
    eps_B = 0.01  # Perturbation amplitude
    v_0 = 10.0  # Set to zero for unstable z-pinch

    # Mesh
    vol = dx**3
    x_lin = jnp.linspace(0.5 * dx, x_length - 0.5 * dx, int(N * x_length))
    y_lin = jnp.linspace(0.5 * dx, x_length - 0.5 * dx, int(N * x_length))
    z_lin = jnp.linspace(0.5 * dx, z_length - 0.5 * dx, int(N * z_length))
    X, Y, Z = jnp.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    cylinder_radius = jnp.sqrt((X - x_length / 2) ** 2 + (Y - x_length / 2) ** 2)
    cylinder_angle = jnp.arctan2(Y - x_length / 2, X - x_length / 2)

    # Generate Initial Conditions
    rho = rho_0 / (1.0 + (cylinder_radius / r_0) ** 2) ** 2
    vx = jnp.zeros(X.shape)
    vy = jnp.zeros(X.shape)
    vz = jnp.zeros(X.shape) + jnp.minimum(v_0 * cylinder_radius / r_0, 100)

    # Get average magnetic field
    B_theta = B_0 * (2.0 * cylinder_radius / r_0) / (1.0 + (cylinder_radius / r_0) ** 2)
    bx = -B_theta * jnp.sin(cylinder_angle)
    by = B_theta * jnp.cos(cylinder_angle)
    bz = jnp.zeros(X.shape)
    Bx, By, Bz = get_B_avg(bx, by, bz)

    # Get initial pressure
    P = initialize_p(Bx, By, Bz, dx, P_0)

    # Add perturbation to magnetic field
    by += eps_B * jnp.cos((Z + X) * 2 * jnp.pi / z_length)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy = get_conserved(
        rho, vx, vy, vz, P, Bx, By, Bz, gamma, vol
    )

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # Simulation Main Loop
    tic = time.time()
    t = 0
    output_counter = 0
    nr_iterations = 0
    save_freq = 0.05
    while t < solve_time:
        # Time step
        (
            bx,
            by,
            bz,
            Mass,
            Momx,
            Momy,
            Momz,
            Energy,
            divB,
            rho,
            dt,
        ) = update(
            bx, by, bz, Mass, Momx, Momy, Momz, Energy, vol, dx, t, gamma, courant_fac
        )

        # determine if we should save the plot
        save_plot = False
        if t + dt > output_counter * save_freq:
            save_plot = True
            output_counter += 1

        # update time
        t += dt

        # update iteration counter
        nr_iterations += 1

        # plot in real time
        if save_plot:
            # Save Image with jet colormap with min/max values
            plt.imsave(
                save_animation_path
                + "/rho_xz_"
                + str(output_counter).zfill(7)
                + ".png",
                np.rot90(rho[::2, X.shape[1] // 2, ::2]),
                cmap="jet",
            )
            plt.imsave(
                save_animation_path
                + "/Momz_xz_"
                + str(output_counter).zfill(7)
                + ".png",
                np.rot90(Momz[::2, X.shape[1] // 2, ::2]),
                cmap="jet",
            )
            plt.imsave(
                save_animation_path + "/bx_" + str(output_counter).zfill(7) + ".png",
                np.rot90(bx[3:-3:2, 0, 3:-3:2]),
                cmap="jet",
            )
            plt.imsave(
                save_animation_path + "/by_" + str(output_counter).zfill(7) + ".png",
                np.rot90(by[3:-3:2, 0, 3:-3:2]),
                cmap="jet",
            )

            # Save VTK file
            write_vtk(
                rho,
                bx,
                by,
                bz,
                filename=save_animation_path
                + "/data_"
                + str(output_counter).zfill(7)
                + ".vtk",
                origin=(0, 0, 0),
                dx=dx,
            )

            # Print progress
            print(
                "Saved state "
                + str(output_counter).zfill(7)
                + " of "
                + str(int(solve_time / save_freq))
                + " at time "
                + str(t)
            )

            # Print million updates per second
            cell_updates = X.shape[0] * X.shape[1] * X.shape[2] * nr_iterations
            total_time = time.time() - tic
            mups = cell_updates / (1e6 * total_time)
            print("Million cell updates per second: ", mups)


if __name__ == "__main__":
    main()
