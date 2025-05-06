"""
Test code adjusted to match (same residuals, same convergence) the FEniCS/Firedrake cases from:
 - https://github.com/nate-sime/dolfin_dg/blob/master/demo/dolfin/compressible_navier_stokes_naca0012/dg_naca0012_2d.py
 - https://github.com/nate-sime/dolfin_dg/blob/master/demo/firedrake/compressible_navier_stokes_naca0012/dg_naca0012_2d.py
"""

import ufl
from math import radians
import numpy as np

try:
	import firedrake
	backend_name = 'firedrake'
except:
	import fenics
	backend_name = 'fenics'

if backend_name == 'firedrake':
	from firedrake import *
	from firedrake.cython import dmcommon
else:
	from dolfin import *

from dolfin_dg import *

import resource
import psutil
import time as time_python
def print_stats(base_time = None):
	p = psutil.Process()
	mem_info = p.memory_full_info()
	current_time = time_python.time()
	if base_time is None:
		elapsed_time = 0.0
	else:
		elapsed_time = current_time - base_time
	peak_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*2**10 # From kB to B
	print("> Time: %1.2f s | RES: %1.2f GB | SHARED: %1.2f GB | SWAP: %1.2f GB | VIRTUAL: %1.2f GB | PEAK RES: %1.2f GB " %(elapsed_time, mem_info.rss/2**30, mem_info.shared/2**30, mem_info.swap/2**30, mem_info.vms/2**30, peak_mem_usage/2**30))
	return current_time
print()
initial_time = print_stats()
print()

if backend_name == 'firedrake':
	parameters['form_compiler']['quadrature_degree'] = 4
else:
	parameters['std_out_all_processes'] = False
	parameters["ghost_mode"] = "shared_facet"
	parameters["form_compiler"]["quadrature_degree"] = 4

if backend_name == 'firedrake':
	base = Mesh("naca0012_coarse_mesh.msh")
else:
	mesh = Mesh()
	XDMFFile("naca0012_coarse_mesh.xdmf").read(mesh)

# Mesh refinement
num_refinements = 5
def refine_mesh(mesh, num_refinements):
	if backend_name == 'firedrake':
		mesh_hierarchy = MeshHierarchy(mesh, refinement_levels = num_refinements)
		mesh = mesh_hierarchy.meshes[num_refinements]
		#File("%s_mesh_test.pvd" %(backend_name)).write(mesh)
	else:
		for i in range(num_refinements):
			mesh = refine(mesh)
		#File("%s_mesh_test.pvd" %(backend_name)) << mesh
	return mesh
if backend_name == 'firedrake':
	pass
else:
	mesh = refine_mesh(mesh, num_refinements)

# Polynomial order
poly_o = 1

# Initial inlet flow conditions
rho_0 = 1.0
M_0 = 0.5
Re_0 = 1e3 # XXX 5e3
p_0 = 1.0
gamma = 1.4
attack = radians(2.0)

# Inlet conditions
c_0 = abs(gamma*p_0/rho_0)**0.5
Re = Re_0/(gamma**0.5*M_0)
if backend_name == 'firedrake':
	n_in = np.array([cos(attack), sin(attack)])
else:
	n_in = Point(cos(attack), sin(attack))
u_ref = Constant(M_0*c_0)
rho_in = Constant(rho_0)
u_in = u_ref*as_vector((Constant(cos(attack)), Constant(sin(attack))))

# Assign variable names to the inlet, outlet and adiabatic wall BCs. These
# indices will be used to define subsets of the boundary.
INLET = 1
OUTLET = 2
WALL = 3

if backend_name == 'firedrake':
	def mark_mesh(mesh):
		dm = mesh.topology_dm
		sec = dm.getCoordinateSection()
		coords = dm.getCoordinatesLocal()
		dm.removeLabel(dmcommon.FACE_SETS_LABEL)
		dm.createLabel(dmcommon.FACE_SETS_LABEL)

		faces = dm.getStratumIS("exterior_facets", 1).indices

		for face in faces:
			vertices = dm.vecGetClosure(sec, coords, face).reshape(2, 2)
			midpoint = vertices.mean(axis=0)

			if midpoint[0]**2 + midpoint[1]**2 < 4:
				dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, WALL)
			else:
				if np.dot(midpoint, n_in) < 0:
					dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, INLET)
				else:
					dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, OUTLET)
		return mesh

	mark_mesh(base)
	mh = MeshHierarchy(base, 0)
	mesh = mh[-1]

else:
	# Label the boundary components of the mesh. Initially label all exterior
	# facets as the adiabatic wall, then label the exterior facets far from
	# the airfoil as the inlet and outlet based on the angle of attack.
	bdry_ff = MeshFunction('size_t', mesh, 1, 0)
	CompiledSubDomain("on_boundary").mark(bdry_ff, WALL)
	for f in facets(mesh):
		x = f.midpoint()
		if not f.exterior() or (x[0]*x[0] + x[1]*x[1]) < 4.0:
			continue
		bdry_ff[f] = INLET if f.normal().dot(n_in) < 0.0 \
			else OUTLET
	ds = Measure('ds', domain=mesh, subdomain_data=bdry_ff)

# Mesh refinement
if backend_name == 'firedrake':
	mesh = refine_mesh(mesh, num_refinements)
else:
	pass

# The initial guess used in the Newton solver. Here we use the inlet flow.
rhoE_in_guess = energy_density(p_0, rho_in, u_in, gamma)
gD_guess = as_vector((rho_in, rho_in*u_in[0], rho_in*u_in[1], rhoE_in_guess))

# Problem function space, (rho, rho*u1, rho*u2, rho*E)
if backend_name == 'firedrake':
	V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4, variant = 'equispaced')
else:
	V = VectorFunctionSpace(mesh, 'DG', poly_o, dim=4)
print("Problem size: %d degrees of freedom" % V.dim())

# Use the initial guess.
if backend_name == 'firedrake':
	u_vec = project(gD_guess, V); u_vec.rename("u")
else:
	u_vec = project(gD_guess, V); u_vec.rename("u", "u")
v_vec = TestFunction(V)

# The subsonic inlet, adiabatic wall and subsonic outlet conditions
inflow = subsonic_inflow(rho_in, u_in, u_vec, gamma)
no_slip_bc = no_slip(u_vec)
outflow = subsonic_outflow(p_0, u_vec, gamma)

# Assemble these conditions into DG BCs
bcs = [DGDirichletBC(ds(INLET), inflow),
       DGAdiabticWallBC(ds(WALL), no_slip_bc),
       DGDirichletBC(ds(OUTLET), outflow)]

# Construct the compressible Navier Stokes DG formulation, and compute
# the symbolic Jacobian
c_ip = 20.0
ce = CompressibleNavierStokesOperator(mesh, V, bcs, mu=1.0/Re)
F = ce.generate_fem_formulation(u_vec, v_vec, c_ip=c_ip)

if backend_name == 'firedrake':
	solver_parameters = {
		"snes_monitor": None,
		"snes_linesearch_maxstep": 1,
		"snes_linesearch_damping": 1,
		"snes_linesearch_monitor": None,
		"ksp_type": "preonly",
		"snes_linesearch_type": "basic",
		"snes_type": "newtonls",
		"pc_type": "lu",
		"pc_factor_mat_solver_type": "mumps",
		'snes_max_it' : 200, 
		'snes_max_funcs' : 2000,
		'snes_atol' : 1E-10,
		'snes_stol' : 1E-10,
		'snes_rtol' : 1e-09,
	}
else:
	PETScOptions.set("snes_monitor")
	PETScOptions.set("snes_linesearch_maxstep", 1)
	PETScOptions.set("snes_linesearch_damping", 1)
	PETScOptions.set("snes_linesearch_monitor")
	PETScOptions.set("ksp_type", "preonly")
	solver_parameters = {
		"nonlinear_solver": "snes",
		"snes_solver": {
			"line_search" : "basic",
			"method" : "newtonls",
			"preconditioner" : "lu",
			"linear_solver": "mumps",
			"maximum_iterations" : 200,
			"maximum_residual_evaluations" : 2000,
			"absolute_tolerance" : 1E-10,
			"solution_tolerance" : 1E-10,
			"relative_tolerance" : 1E-9,
			"error_on_nonconvergence" : True,
		}
	}

print()
print("Solve")
print_stats(initial_time)
solve(F == 0, u_vec, solver_parameters = solver_parameters)
print_stats(initial_time)
print()

total_solves = 3
for i in range(total_solves):

	print()
	print("Solve %d/%d" %(i + 1, total_solves))
	print_stats(initial_time)

	c_ip = 20.0 + (i + 1)*10.
	ce = CompressibleNavierStokesOperator(mesh, V, bcs, mu=1.0/Re)
	F = ce.generate_fem_formulation(u_vec, v_vec, c_ip=c_ip)

	solve(F == 0, u_vec, solver_parameters = solver_parameters)

	print_stats(initial_time)
	print()

print("Enforced garbage collection...")
import gc
gc.collect() 
import ctypes
def malloc_trim():
	ctypes.CDLL('libc.so.6').malloc_trim(0)
malloc_trim()
print_stats(initial_time)
print()
