from thesis.Elliptic.__init__ import *
import time

RUNSIMU = False
EXTENSION = ".pdf"
degree, cuts = 6, 6

if RUNSIMU:
	with open(f"{FOLDER2DATA}preconditioner_all_output.txt", "w") as file:
		for preconditioner in ["WP", "FD", "ILU", "JM"]:
			problem = simulate_ht(
				degree, cuts,
				preconditioner=preconditioner,
				linsolver="GMRES",
			)[0]
			residue = problem._linear_residual_list[0]
			file.write(f"{residue}\n")

	with open(f"{FOLDER2DATA}preconditioner_JM_output.txt", "w") as file:
		file.write("Starting simulation...\n")
		for degree in range(4, 7):
			for cuts in range(6, 10):
				start = time.process_time()
				problem = simulate_ht(degree, cuts, preconditioner="JM")[0]
				residue = problem._linear_residual_list[0]
				stop = time.process_time()
				file.write(
					"%d, %d, %.2f, %d \n" % (degree, cuts, stop - start, len(residue))
				)
		file.write("Simulation complete.\n")

with open(f"{FOLDER2DATA}preconditioner_all_output.txt", 'r') as archivo:
	lineas = archivo.readlines()

results_list = []
for i, linea in enumerate(lineas):
	results_list.append(np.array(list(map(float, linea[1:-2].strip().split(',')))))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for j, [preconditioner, labelfig] in enumerate(zip(["WP", "FD", "ILU", "JM"], 
												['w.o. preconditioner', 
												'Standard Fast Diag.',
												'Incomplete LU', 
												'Our contribution'])):

	ax.semilogy(results_list[j], label=labelfig, markevery=5,
			marker=MARKERLIST[j], color=COLORLIST[j])

ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.2), loc="upper center")
ax.set_ylabel("Relative residue")
ax.set_xlabel("Number of iterations (GMRES)")
ax.set_ylim([1e-12, 1e1])
ax.set_xlim([0, 100])
fig.savefig(f"{FOLDER2RESU}performance_preconditioner{EXTENSION}")
