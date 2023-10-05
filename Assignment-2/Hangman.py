import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your decision tree model using the word list provided
	# Return the trained model as is -- do not compress it using pickle etc
	# Model packing or compression will cause evaluation failure
	
	model = Tree( min_leaf_size = 1, max_depth = 15 )
	model.fit( words )

	return model					# Return the trained model

class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words, verbose = False ):
		self.words = words
		self.root = Node( depth = 0, parent = None )
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth )


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.history = []
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- just return the first word
	def process_leaf( self, my_words_idx, history ):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query

	def get_entropy( self, counts ):    
		assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
		num_elements = counts.sum()
		if num_elements <= 1:
			print( f"warning: { num_elements } elements in total." )
			return 0
		proportions = counts / num_elements
		return np.sum( proportions * np.log2( counts ) )  

	def try_attr( self, query, my_words_idx, all_words ):
		split_dict = {}
		count_dict = {}
			
		for idx in my_words_idx:
			mask = self.reveal( all_words[ idx ], all_words[query] )
			if mask not in split_dict:
				split_dict[ mask ] = []
				count_dict[ mask ] = 0
			
			split_dict[ mask ].append( idx )
			count_dict[ mask ] += 1

		entropy = self.get_entropy( np.array( list( count_dict.values() ) ) )
		
		return ( entropy, split_dict )

	def get_split_actor( self, query_list, my_words_idx, all_words ):
		
		best_entropy = np.inf
		best_query = -1
		best_split_dict = None
		
		for query in query_list:
			entropy, split_dict = self.try_attr( query, my_words_idx, all_words )
	#         print( f"{c}({entropy})", end = ", " )
			if entropy < best_entropy:
				best_entropy = entropy
				best_query = query
				best_split_dict = split_dict
		
	#     print()
	#     print( f"-->{best_attr}({best_entropy})\n" )
		
		return ( best_query , best_split_dict )

	def process_node( self, all_words, my_words_idx, history ):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if len( history ) == 0:
			query_idx = -1
			query = ""
			split_dict = {}
			for idx in my_words_idx:
				mask = self.reveal( all_words[ idx ], query )
				if mask not in split_dict:
					split_dict[ mask ] = []
				split_dict[ mask ].append( idx )

		else:
			num_query = int(np.log2(len(my_words_idx)))
			query_list = np.random.choice(my_words_idx,num_query)
			query_idx, split_dict = self.get_split_actor(query_list, my_words_idx, all_words)

		if len( split_dict.items() ) < 2:
			print( "Warning: did not make any meaningful split with this query!" )
		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    " ):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx, self.history )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				history = self.history.copy()
				history.append( [ self.query_idx, response ] )
				self.children[ response ].history = history
				
				# Recursively train this child node
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str )