from .__init__ import *
from typing import List, Set, Dict

def flatten_list(list_in:list, offset:int):
	list_out = [x + i * offset for i, sublist in enumerate(list_in) for x in sublist]
	return list_out

def convert_boundary_str_to_int(dir_list:list, fc_list:list):
	dir_map = {'x': 0, 'y': 1, 'z': 2, 't':3,
				'0': 0, '1': 1, '2': 2, '3': 3,}
	face_map = {'left': 0, 'front': 0, 'bottom': 0,
				'right': 1, 'back': 1, 'top': 1}
	dir_list_seperated = re.split(r'[;,\|\s]+', dir_list)
	fc_list_seperated = re.split(r'[;,\|\s]+', fc_list)
	output_dir = [[dir_map.get(d)] for d in dir_list_seperated]
	output_facemap = [[0, 1] if f == 'both' else [face_map.get(f)] for f in fc_list_seperated]
	return output_dir, output_facemap

def create_connectivity_table(nnz_by_direction:np.ndarray):
	indices = np.reshape(
		np.indices(nnz_by_direction), 
		newshape=(len(nnz_by_direction), -1),
	)
	return np.transpose(indices).astype(int)

class boundary_condition():
	def __init__(self, nbctrlpts:np.ndarray=np.array([1, 1, 1]), nbvars:int=1):
		self.nbvars:int = nbvars
		self._nbctrlpts:np.ndarray = nbctrlpts[nbctrlpts>0]
		self._nbctrlpts_total:int = np.product(self._nbctrlpts)
		self._connectivity_table:np.ndarray = create_connectivity_table(self._nbctrlpts)
		self.ctrlpts_dirichlet:List[Set] = [set() for _ in range(self.nbvars)]
		self.table_dirichlet:np.ndarray = np.zeros(
			shape=(nbvars, np.max([3, len(nbctrlpts)]), 2), 
			dtype=bool
		)

	def _recognize_constraint(self, location_list:List[Dict]):
		table = np.zeros_like(self.table_dirichlet, dtype=bool)
		nodes = [set() for _ in range(self.nbvars)]
		for kk, loc in enumerate(location_list):
			loc_dir_list = str(loc.get('direction')).lower()
			loc_fc_list = str(loc.get('face')).lower()
			idx_dir_list, idx_fc_list = convert_boundary_str_to_int(loc_dir_list, loc_fc_list)
			for idx_dir, idx_fc in zip(idx_dir_list, idx_fc_list):
				table[kk, idx_dir, idx_fc] = True
				for ii in idx_dir:
					for jj in idx_fc: 
						if jj == 0:	
							nodes[kk].update(np.where(self._connectivity_table[:, ii] == 0)[0])
						else:
							nodes[kk].update(np.where(self._connectivity_table[:, ii] == self._nbctrlpts[ii] - 1)[0])
		return nodes, table
	
	def add_constraint(self, location_list:List[Dict], constraint_type:str):
		assert len(location_list) >= self.nbvars, 'The number of conditions is not well defined'
		nodes, table = self._recognize_constraint(location_list)
		if constraint_type.lower() == "dirichlet":
			for _ in range(self.nbvars):
				self.ctrlpts_dirichlet[_].update(nodes[_])
			self.table_dirichlet = np.logical_or(self.table_dirichlet, table)
		# TODO: Add contact conditions here
		else: 
			raise ValueError("Type invalid")
		return

	def select_nodes4solving(self):
		if not self.ctrlpts_dirichlet: print('The set of Dirichlet conditions is empty')
		free_nodes = [[] for _ in range(self.nbvars)]
		constraint_nodes = [[] for _ in range(self.nbvars)]
		for _ in range(self.nbvars):
			free_nodes[_] = sorted(list(set(range(self._nbctrlpts_total)).difference(self.ctrlpts_dirichlet[_])))
			constraint_nodes[_] = sorted(list(self.ctrlpts_dirichlet[_])) 
		return free_nodes, constraint_nodes