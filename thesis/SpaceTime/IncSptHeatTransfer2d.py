from thesis.SpaceTime.__init__ import *
from thesis.SpaceTime.input_data import *
import time

def exactTemperature_inc(args):
	func = exactTemperatureRing_inc(args)
	return func

def exactTemperature_spt(args):
	func = exactTemperatureRing_spt(args)
	return func

def powerDensity_inc(args):
	func = powerDensityRing_inc(args)
	return func

def powerDensity_spt(args):
	func = powerDensityRing_spt(args)
	return func

# Set global variables
PLOTRELATIVE = True
RUNSIMU = False
FIG_CASE = 5
EXTENSION = '.dat'

if FIG_CASE == 1:

	pass

	if RUNSIMU:
		degree_list = np.array([1, 2, 3, 4, 5])
		cut_list = np.arange(1, 6)
		for quadrule, quadtype in zip(['gs', 'wq'], ['leg', 2]):
			sufix = f"_{quadrule}_{str(quadtype)}"
			quad_args = {'quadrule': quadrule, 'type': quadtype}
			abserror_list = np.zeros((len(degree_list)+1, len(cut_list)+1))
			relerror_list = np.zeros((len(degree_list)+1, len(cut_list)+1))
			abserror_list[0, 1:] = cut_list; relerror_list[0, 1:] = cut_list
			abserror_list[1:, 0] = degree_list; relerror_list[1:, 0] = degree_list
			filenameA1 = FOLDER2DATA+'1l2abserror'+sufix+EXTENSION
			filenameR1 = FOLDER2DATA+'1l2relerror'+sufix+EXTENSION
			for j, cuts in enumerate(cut_list):
				for i, degree in enumerate(degree_list):
					problem_spt, time_spt, temp_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
													quad_args=quad_args, degree_time=degree, nbel_time=2**cuts)
					abserror_list[i+1, j+1], relerror_list[i+1, j+1] = problem_spt.norm_of_error(temp_spt,
																	norm_args={'type':'L2',
																	'exact_function':exactTemperature_spt},)
					np.savetxt(filenameA1, abserror_list)
					np.savetxt(filenameR1, relerror_list)

	from mpltools import annotation
	plotoptions = [CONFIGLINE_IGA, CONFIGLINE_WQ]
	figname = f"{FOLDER2RESU}1_L2Convergence.pdf"
	if PLOTRELATIVE: filenames = ['1l2relerror_gs_leg', '1l2relerror_wq_2']
	else: filenames = ['1l2abserror_gs_leg', '1l2abserror_wq_2']

	fig, ax = plt.subplots(figsize=(5, 5))
	for filename, plotopts in zip(filenames, plotoptions):
		quadrule = filename.split('_')[1]
		table = np.loadtxt(f"{FOLDER2DATA}{filename}{EXTENSION}")
		nbel_list = 2**(table[0, 1:])
		degree_list = table[1:, 0]
		error_list  = table[1:, 1:]
		for i, degree in enumerate(degree_list):
			color = COLORLIST[i]
			if quadrule == 'gs':
				ax.loglog(nbel_list, error_list[i, :], label=fr'ST-IGA-GL $p={int(degree)}$', color=color, **plotopts)
				slope = np.polyfit(np.log10(nbel_list[2:]),np.log10(error_list[i, 2:]), 1)[0]
				slope = int(round(slope, 1))
				annotation.slope_marker((nbel_list[-2], error_list[i, -2]), slope,
								poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

			else:
				ax.loglog(nbel_list, error_list[i, :], color=color, **plotopts)
			fig.savefig(figname)
	ax.loglog([], [], color='k', **CONFIGLINE_WQ, label='ST-IGA-WQ')

	if PLOTRELATIVE:
		ax.set_ylabel(r'Relative $L^2(\Pi)$ error')
		ax.set_ylim(top=1e0, bottom=1e-9)
	else:
		ax.set_ylabel(r'$L^2(\Pi)$ error')
		ax.set_ylim(top=1e1, bottom=1e-8)

	ax.set_xlabel('Number of elements by space-time direction')
	ax.set_xlim(left=1, right=50)
	ax.legend(loc='lower left')
	fig.tight_layout()
	fig.savefig(figname)

elif FIG_CASE == 2:

	pass

	degree, cuts = 8, 6
	quad_args = {'quadrule':'wq', 'type':2}
	nbel_time_list = np.array([2**cuts for cuts in range(2, 7)], dtype=int)
	degree_time_list = np.arange(1, 5)
	abserror_inc, relerror_inc = np.ones(len(nbel_time_list)), np.ones(len(nbel_time_list))
	abserror_inc2, relerror_inc2 = np.ones(len(nbel_time_list)), np.ones(len(nbel_time_list))
	abserror_spt, relerror_spt = np.ones((len(degree_time_list), len(nbel_time_list))), np.ones((len(degree_time_list), len(nbel_time_list)))
	abserror_spt2, relerror_spt2 = np.ones((len(degree_time_list), len(nbel_time_list))), np.ones((len(degree_time_list), len(nbel_time_list)))

	if RUNSIMU:
		for i, nbel_time in enumerate(nbel_time_list):

			problem_spt_inc = simulate_spacetime(degree, cuts, powerDensity_spt,
												degree_time=1, nbel_time=nbel_time,
												quad_args={'quadrule':'gs'}, solve_system=False)[0]

			problem_inc, time_inc, temp_inc = simulate_incremental(degree, cuts, powerDensity_inc,
														nbel_time=nbel_time, quad_args=quad_args)

			abserror_inc[i], relerror_inc[i] = problem_spt_inc.norm_of_error(np.ravel(temp_inc, order='F'),
										norm_args={'type':'L2',
												'exact_function':exactTemperature_spt})

			abserror_inc2[i], relerror_inc2[i] = problem_inc.norm_of_error(temp_inc[:, -1],
										norm_args={'type':'L2',
												'exact_function':exactTemperature_inc,
												'exact_args':{'time':time_inc[-1]}})

			np.savetxt(FOLDER2DATA+'2abserrorstag_inc'+EXTENSION, abserror_inc)
			np.savetxt(FOLDER2DATA+'2relerrorstag_inc'+EXTENSION, relerror_inc)
			np.savetxt(FOLDER2DATA+'2abserrorstag_inc2'+EXTENSION, abserror_inc2)
			np.savetxt(FOLDER2DATA+'2relerrorstag_inc2'+EXTENSION, relerror_inc2)

			for j, degree_spt in enumerate(degree_time_list):

				problem_spt, time_spt, temp_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
													degree_time=degree_spt, nbel_time=nbel_time, quad_args=quad_args)

				abserror_spt[j, i], relerror_spt[j, i] = problem_spt.norm_of_error(temp_spt,
														norm_args={'type':'L2',
																'exact_function':exactTemperature_spt,})

				abserror_spt2[j, i], relerror_spt2[j, i] = problem_inc.norm_of_error(np.reshape(temp_spt, order='F',
														newshape=(problem_spt.part.nbctrlpts_total, time_spt.nbctrlpts_total))[:, -1],
														norm_args={'type':'L2',
																'exact_function':exactTemperature_inc,
																'exact_args':{'time':time_inc[-1]}})

				np.savetxt(FOLDER2DATA+'2abserrorstag_spt'+EXTENSION, abserror_spt)
				np.savetxt(FOLDER2DATA+'2relerrorstag_spt'+EXTENSION, relerror_spt)
				np.savetxt(FOLDER2DATA+'2abserrorstag_spt2'+EXTENSION, abserror_spt2)
				np.savetxt(FOLDER2DATA+'2relerrorstag_spt2'+EXTENSION, relerror_spt2)

	from mpltools import annotation
	fig, ax = plt.subplots(figsize=(5, 5))

	if PLOTRELATIVE: errorList1 = np.loadtxt(FOLDER2DATA+'2relerrorstag_spt'+EXTENSION)
	else: errorList1 = np.loadtxt(FOLDER2DATA+'2abserrorstag_spt'+EXTENSION)
	for i, deg in enumerate(degree_time_list):
		nbctrlpts = nbel_time_list+deg
		ax.loglog(nbctrlpts, errorList1[i, :], color=COLORLIST[i], **CONFIGLINE_IGA, label=fr'ST-IGA-GL $p_t={int(deg)}$')
		slope = np.polyfit(np.log10(nbctrlpts[3:]),np.log10(errorList1[i, 3:]), 1)[0]
		slope = round(slope, 1)
		annotation.slope_marker((nbctrlpts[-2], errorList1[i, -2]), slope,
						poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

	if PLOTRELATIVE: errorList1 = np.loadtxt(FOLDER2DATA+'2relerrorstag_inc'+EXTENSION)
	else: errorList1 = np.loadtxt(FOLDER2DATA+'2abserrorstag_inc'+EXTENSION)

	nbctrlpts = nbel_time_list+1
	ax.loglog(nbctrlpts, errorList1, color='k', **CONFIGLINE_INC, label='Crank-Nicolson')
	slope = np.polyfit(np.log10(nbctrlpts[3:]),np.log10(errorList1[3:]), 1)[0]
	slope = round(slope, 1)
	annotation.slope_marker((nbctrlpts[-2], errorList1[-2]), slope,
					poly_kwargs={'facecolor': (0.73, 0.8, 1)}, ax=ax)

	if PLOTRELATIVE:
		ax.set_ylabel(r'Relative $L^2(\Pi)$ error')
		ax.set_ylim(top=1e0, bottom=1e-12)
	else:
		ax.set_ylabel(r'$L^2(\Pi)$ error')
		ax.set_ylim(top=1e1, bottom=1e-8)

	ax.set_xlabel('Number of control points in time \n(or number of time-steps)')
	ax.set_xlim(left=2, right=100)
	ax.legend(loc='lower left')
	fig.tight_layout()
	fig.savefig(f"{FOLDER2RESU}2_StagnationError.pdf")

if FIG_CASE == 3:

	pass

	filenameA2 = FOLDER2DATA + '3incheatAbs'
	filenameR2 = FOLDER2DATA + '3incheatRel'
	filenameT2 = FOLDER2DATA + '3incheatTim'

	filenameA3 = FOLDER2DATA + '3sptheatAbs'
	filenameR3 = FOLDER2DATA + '3sptheatRel'
	filenameT3 = FOLDER2DATA + '3sptheatTim'

	degree_list = np.array([1, 2, 3, 4, 5, 6])
	cuts_list = np.arange(4, 7)

	if RUNSIMU:

		A2errorList = np.ones((len(degree_list), len(cuts_list)))
		R2errorList = np.ones((len(degree_list), len(cuts_list)))
		T2timeList = np.ones((len(degree_list), len(cuts_list)))
		quadArgs = {'quadrule': 'wq', 'type': 2}
		sufix = f"_wq_2{EXTENSION}"
		for j, cuts in enumerate(cuts_list):
			for i, degree in enumerate(degree_list):

				problem_spt, time_spt, temp_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
													degree_time=1, nbel_time=2**cuts, quad_args=quadArgs,
													auto_inner_tolerance=False, solve_system=False)
				start = time.process_time()
				problem_inc, time_inc, temp_inc = simulate_incremental(degree, cuts, powerDensity_inc, nbel_time=2**cuts)
				finish = time.process_time()
				T2timeList[i, j] = finish - start

				A2errorList[i, j], R2errorList[i, j] = problem_spt.norm_of_error(np.ravel(temp_inc, order='F'),
														norm_args={'type':'L2',
																'exact_function':exactTemperature_spt,})

				np.savetxt(filenameA2+sufix, A2errorList)
				np.savetxt(filenameR2+sufix, R2errorList)
				np.savetxt(filenameT2+sufix, T2timeList)

		A3errorList = np.ones((len(degree_list), len(cuts_list)))
		R3errorList = np.ones((len(degree_list), len(cuts_list)))
		T3timeList = np.ones((len(degree_list), len(cuts_list)))
		for quadrule, quadtype in zip(['wq', 'gs'], [2, 'leg']):
			quadArgs = {'quadrule': quadrule, 'type': quadtype}
			sufix = f"_{quadrule}_{quadtype}{EXTENSION}"
			for j, cuts in enumerate(cuts_list):
				for i, degree in enumerate(degree_list):

					start = time.process_time()
					problem_spt, time_spt, temp_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
														degree_time=degree, nbel_time=2**cuts,
														quad_args=quadArgs, auto_inner_tolerance=False)

					end = time.process_time()
					T3timeList[i, j] = end - start

					A3errorList[i, j], R3errorList[i, j] = problem_spt.norm_of_error(temp_spt,
															norm_args={'type':'L2',
																	'exact_function':exactTemperature_spt,})

					np.savetxt(filenameA3+sufix, A3errorList)
					np.savetxt(filenameR3+sufix, R3errorList)
					np.savetxt(filenameT3+sufix, T3timeList)

	position = 2
	assert position in [1, 2], 'Must be one or 2'
	fig, ax = plt.subplots(figsize=(6.5, 5.5))
	cmap = mpl.colors.ListedColormap(COLORLIST[:len(degree_list)])

	if PLOTRELATIVE:
		filenameA2 = FOLDER2DATA + '3incheatRel'
		filenameA3 = FOLDER2DATA + '3sptheatRel'

	Elist = np.loadtxt(f"{filenameA2}_wq_2{EXTENSION}")
	Tlist = np.loadtxt(f"{filenameT2}_wq_2{EXTENSION}")
	im = ax.scatter(Tlist[:len(degree_list), position-1], Elist[:len(degree_list), position-1], 
			cmap=cmap, c=degree_list, facecolors=CONFIGLINE_INC['markerfacecolor'], 
			marker=CONFIGLINE_INC['marker'], s=15*CONFIGLINE_INC['markersize'], 
			)
	ax.loglog(Tlist[:len(degree_list), position-1], Elist[:len(degree_list), position-1],
			color='k', marker='', linestyle=CONFIGLINE_INC['linestyle'])

	cbar = fig.colorbar(im); cbar.set_label('Degree')
	tick_locs = 1+(np.arange(len(degree_list)) + 0.5)*(len(degree_list)-1)/len(degree_list)
	cbar.set_ticks(tick_locs)
	cbar.set_ticklabels(degree_list)

	for quadrule, quadtype, plotopts in zip(['gs', 'wq'], ['leg', 2], [CONFIGLINE_IGA, CONFIGLINE_WQ]):
		sufix = f"_{quadrule}_{quadtype}{EXTENSION}"
		Elist = np.loadtxt(filenameA3+sufix)
		Tlist = np.loadtxt(filenameT3+sufix)
		ax.scatter(Tlist[:len(degree_list), position-1], Elist[:len(degree_list), position-1], 
					cmap=cmap, c=degree_list, facecolors=plotopts['markerfacecolor'],
					marker=plotopts['marker'], s=15*plotopts['markersize'])

		ax.loglog(Tlist[:len(degree_list), position-1], Elist[:len(degree_list), position-1],
				color='k', marker='', linestyle=plotopts['linestyle'])

	ax.loglog([], [], color='k', alpha=0.5, **CONFIGLINE_INC, label='Crank-Nicolson')
	ax.loglog([], [], color='k', alpha=0.5,  **CONFIGLINE_IGA, label='ST-IGA-GL')
	ax.loglog([], [], color='k', alpha=0.5,  **CONFIGLINE_WQ, label='ST-IGA-WQ')

	if PLOTRELATIVE:
		ax.set_ylabel(r'Relative $L^2(\Pi)$ error')
		ax.set_ylim(top=1e-1, bottom=1e-8)
	else:
		ax.set_ylabel(r'$L^2(\Pi)$ error')
		ax.set_ylim(top=1e-1, bottom=1e-11)

	ax.set_xlabel('CPU time (s)')
	ax.legend()
	if position==1: ax.set_xlim(left=1e-1, right=1e2)
	if position==2:
		ax.set_xlim(left=1e0, right=1e3)
		ax.set_ylim(top=1e-2, bottom=1e-10)
	fig.tight_layout()
	fig.savefig(f'{FOLDER2RESU}3_SPTINC_CPUError_{position}.pdf')

elif FIG_CASE == 4:
	pass

	degree, cuts = 3, 4
	subfolderfolder = FOLDER2DATA + str(degree) + '_' + str(cuts) + '/'
	if not os.path.isdir(subfolderfolder): os.mkdir(subfolderfolder)

	if RUNSIMU:
		for [i, is_adaptive], prefix1 in zip(enumerate([False, True]), ['exact', 'inexact']):
			for [j, isnewton], prefix2 in zip(enumerate([True, False]), ['newton', 'picard']):
				prefix = prefix1 + '_' + prefix2 + '_'

				start = time.process_time()
				problem_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
												degree_time=degree, nbel_time=2**cuts,
												auto_inner_tolerance=is_adaptive, use_newton=isnewton)[0]
				stop = time.process_time()
				print("Method %s %s %.3f" % (prefix1, prefix2, stop-start))

				resKrylov_list = problem_spt._linear_residual_list
				resNewton_list = problem_spt._nonlinear_residual_list

				resKrylovclean = np.array([]); counter_list = [0]
				for krylist in resKrylov_list:
					resKrylovclean = np.append(resKrylovclean, krylist)
					counter_list.append(counter_list[-1] + len(krylist))

				np.savetxt(subfolderfolder+prefix+'CumulKrylovRes'+EXTENSION, resKrylovclean)
				np.savetxt(subfolderfolder+prefix+'Inner_loops'+EXTENSION, counter_list)
				np.savetxt(subfolderfolder+prefix+'NewtonRes'+EXTENSION, resNewton_list)

	fig1, ax1 = plt.subplots(figsize=(5, 5))
	fig2, ax2 = plt.subplots(figsize=(5, 5))
	figs = [fig1, fig2]; axs  = [ax1, ax2]
	linestyle_list = ['-', '--', '-', '--']
	marker_list = ['o', 'o', 's', 's']

	for [i, is_adaptive], prefix1 in zip(enumerate([True, False]), ['inexact', 'exact']):
		for [j, isnewton], prefix2 in zip(enumerate([True, False]), ['newton', 'picard']):
			l = j + i*2
			legendname = prefix1.capitalize() + ' ' + prefix2.capitalize()
			prefix = prefix1 + '_' + prefix2 + '_'
			nbInnerLoops = np.loadtxt(subfolderfolder+prefix+'Inner_loops'+EXTENSION)
			newtonRes = np.loadtxt(subfolderfolder+prefix+'NewtonRes'+EXTENSION)
			newtonRes = newtonRes/newtonRes[0]

			for caseplot, fig, ax in zip(range(1, 3), figs, axs):
				ylim = 1e-11
				if caseplot == 1:
					yy = newtonRes; xx = nbInnerLoops[:len(newtonRes)]
					xlim = 200
					ylabel = 'Relative norm of nonlinear residue'
					xlabel = 'Number of matrix-vector products'
				elif caseplot == 2:
					yy = newtonRes; xx = np.arange(0, len(newtonRes))
					xlim = 10
					ylabel = 'Relative norm of nonlinear residue'
					xlabel = 'Number of nonlinear iterations'

				ax.semilogy(xx, yy, label=legendname, marker=marker_list[l], linestyle=linestyle_list[l])
				ax.set_xlim(right=xlim, left=0)
				ax.set_ylim(top=1e1, bottom=ylim)
				ax.set_xlabel(xlabel)
				ax.set_ylabel(ylabel)
				if caseplot==3 or caseplot==4: ax.legend()
				fig.tight_layout()
				fig.savefig(f"{FOLDER2RESU}4_NLConvergence_iters_{degree}_{cuts}_{caseplot}.pdf")

elif FIG_CASE == 5:

	degList = np.array([1, 2, 3, 4, 5, 6, 7, 8])
	cutList = np.arange(2, 7)

	filenameA3 = FOLDER2DATA + '5sptheatAbs'
	filenameR3 = FOLDER2DATA + '5sptheatRel'
	filenameT3 = FOLDER2DATA + '5sptheatTim'

	if RUNSIMU:

		quadArgs = {'quadrule': 'wq', 'type': 2}
		A3errorList = np.ones((len(degList), len(cutList)))
		R3errorList = np.ones((len(degList), len(cutList)))
		T3timeList = np.ones((len(degList), len(cutList)))

		for quadrule, quadtype in zip(['wq'], [2]):
			quadArgs = {'quadrule': quadrule, 'type': quadtype}
			sufix = f'_{quadrule}_{quadtype}{EXTENSION}'
			for j, cuts in enumerate(cutList):
				for i, degree in enumerate(degList):

					start = time.process_time()
					problem_spt, time_spt, temp_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
														degree_time=degree, nbel_time=2**cuts, quad_args=quadArgs,
														auto_inner_tolerance=False)

					end = time.process_time()
					T3timeList[i, j] = end - start

					A3errorList[i, j], R3errorList[i, j] = problem_spt.norm_of_error(temp_spt,
															norm_args={'type':'L2',
																	'exact_function':exactTemperature_spt,})

					np.savetxt(filenameA3+sufix, A3errorList)
					np.savetxt(filenameR3+sufix, R3errorList)
					np.savetxt(filenameT3+sufix, T3timeList)

	fig2, ax2 = plt.subplots(figsize=(5.5, 4))
	cmap = plt.get_cmap('RdYlGn', 8)

	for quadrule, quadtype, plotopts, ax in zip(['wq'], [2], [CONFIGLINE_WQ], [ax2]):
		sufix = f'_{quadrule}_{quadtype}{EXTENSION}'
		Elist = np.loadtxt(filenameA3+sufix)
		Tlist = np.loadtxt(filenameT3+sufix)
		for pos in range(np.size(Elist, axis=1)):
			im = ax.scatter(Tlist[:len(degList), pos], Elist[:len(degList), pos], c=degList,
							cmap=cmap, marker=plotopts['marker'], s=15*plotopts['markersize'])

			ax.loglog(Tlist[:len(degList), pos], Elist[:len(degList), pos],
					color='k', marker='', linestyle=plotopts['linestyle'], alpha=0.3)
			ax.text(Tlist[-1, pos]*0.5, Elist[-1, pos]/32, str(int(2**(pos+2)))+r'$^3$'+' el.')

		cbar = plt.colorbar(im, ax=ax)
		cbar.set_label('Degree')
		tick_locs = 1+(np.arange(len(degList)) + 0.5)*(len(degList)-1)/len(degList)
		cbar.set_ticks(tick_locs)
		cbar.set_ticklabels(degList)

	for fig, ax, sufix in zip([fig2], [ax2], ['WQ']):
		ax.grid(False)
		ax.set_ylabel(r'Relative $L^2(\Pi)$ error')
		ax.set_ylim(top=1e0, bottom=1e-16)
		ax.set_xlim(left=1e-1, right=1e3)
		ax.set_xlabel('CPU time (s)')
		fig.tight_layout()
		fig.savefig(f'{FOLDER2RESU}5_SPTINC_CPUError{sufix}.pdf')

elif FIG_CASE == 6:
	time_list = [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1]

	degree, cuts = 8, 3
	problem_spt = simulate_spacetime(degree, cuts, powerDensity_spt,
									degree_time=degree, nbel_time=2**cuts, 
									solve_system=False)[0]
	for i, t in enumerate(time_list):
		problem_spt.part.postprocessing_primal(fields={'temp':exactTemperature_inc}, 
											folder=FOLDER2RESU, name=f'out_{i}', 
											extra_args={'time':t, 
											'temperature':exactTemperature_inc})
	
	import imageio
	from src.lib_part import vtk2png
	images = []
	for i in range(len(time_list)):
		filepath = vtk2png(folder=FOLDER2RESU, filename=f'out_{i}', fieldname='temp', clim=(0, 15),
							cmap='coolwarm', title='Temperature', camera_position='xy')
		images.append(imageio.imread(filepath))
	images.append(imageio.imread(filepath))
	images.append(imageio.imread(filepath))
	images = images[-1::-1] + images
	imageio.mimsave(f"{FOLDER2RESU}temperature.gif", images)
