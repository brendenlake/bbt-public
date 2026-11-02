import numpy as np
import random
try:
    from LOTlib3.Grammar import Grammar
    from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
    from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
    from LOTlib3.DataAndObjects import Obj # for nicely managing data
except ModuleNotFoundError as exc:
    if exc.name != 'LOTlib3':
        raise
    raise ModuleNotFoundError(
        'Logical-concept Bayesian workflows require LOTlib3. Clone '
        'https://github.com/piantado/LOTlib3 and add its parent directory '
        'to PYTHONPATH. See "LOTlib dependency" in README.md.',
        name='LOTlib3',
    ) from exc

## Boolean concept learning with sets of examples, from Piantadosi et al. Psych Review paper

symb_shapes = ['circle','triangle','rectangle']
symb_colors = ['blue','yellow','green']
symb_sizes = ['size_1','size_2','size_3']
symb_special = ['OBJ_SEP']

grammar = Grammar()

## Boolean logic
grammar.add_rule('START', 'False', None, 1.0)
grammar.add_rule('START', 'True', None, 1.0)
grammar.add_rule('START', '', ['BOOL'], 10.0)

# boolean rules
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
grammar.add_rule('BOOL', '', ['PREDICATE'], 10.0)  # Upweight to make well-formed
grammar.add_rule('PREDICATE', 'is_color_',  ['OBJECT', '\'blue\''],   1.0)
grammar.add_rule('PREDICATE', 'is_color_',  ['OBJECT', '\'yellow\''],  1.0)
grammar.add_rule('PREDICATE', 'is_color_',  ['OBJECT', '\'green\''], 1.0)
grammar.add_rule('PREDICATE', 'is_shape_',  ['OBJECT', '\'circle\''],   1.0)
grammar.add_rule('PREDICATE', 'is_shape_',  ['OBJECT', '\'triangle\''],  1.0)
grammar.add_rule('PREDICATE', 'is_shape_',  ['OBJECT', '\'rectangle\''], 1.0)
grammar.add_rule('PREDICATE', 'is_size_',  ['OBJECT', '\'size_1\''],   1.0)
grammar.add_rule('PREDICATE', 'is_size_',  ['OBJECT', '\'size_2\''],  1.0)
grammar.add_rule('PREDICATE', 'is_size_',  ['OBJECT', '\'size_3\''], 1.0)

grammar.add_rule('OBJECT', 'x',  None,   1.0)

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display='lambda x: %s', **kwargs)

def x_to_fdata(x):
     # Covert standard format for "xs" or 'xq' into list of LOTLib objects
    # Input
    # x : two possible formats for a set of objects:
    #   1) short format: ['circle_blue_size3','circle_blue_size3']
    #   2) long format: ['triangle', 'yellow', 'size_1', 'OBJ_SEP', 'circle', 'blue', 'size_3'] or ['rectangle', 'green', 'size_2']
    # Output:
    #  list_of_Obj : list of objects, each is an Object class
    if len(x) == 0:
        return []
    elif get_format_x(x) == 'long':
        return helper_x_to_fdata_long(x)
    elif get_format_x(x) == 'short':
        return helper_x_to_fdata_short(x)
    else:
        raise ValueError(f"Unrecognized format for x: {x}")

def get_format_x(x):
    # Input
    # x : two possible formats for a set of objects:
    #   1) short format: ['circle_blue_size3','circle_blue_size3']
    #   2) long format: ['triangle', 'yellow', 'size_1', 'OBJ_SEP', 'circle', 'blue', 'size_3'] or ['rectangle', 'green', 'size_2']
    if x[0] in symb_shapes and x[1] in symb_colors and x[2] in symb_sizes:
        return 'long'
    token1 = x[0]
    obj1 = x[0].split('_')
    if len(obj1) != 3:
        raise ValueError(f"Unrecognized format for x: {x}")
    if obj1[0] in symb_shapes and obj1[1] in symb_colors and obj1[2].replace('size', 'size_') in symb_sizes:
        return 'short'
    raise ValueError(f"Unrecognized format for x: {x}")

def helper_x_to_fdata_short(x):
    # format: ['circle_blue_size3','circle_blue_size3']
    list_of_Obj = []
    for myobj in x:
        item = myobj.strip().split('_')
        assert(len(item)==3)
        assert('size' in item[2])
        list_of_Obj.append(Obj(shape=item[0], color=item[1], size=item[2].replace('size', 'size_')))
    return list_of_Obj

def helper_x_to_fdata_long(x):    
    # format x : list of symbols, with different objects separated as OBJ_SEP ['triangle', 'yellow', 'size_1', 'OBJ_SEP', 'circle', 'blue', 'size_3']
    x_str = ' '.join(x)
    items_as_lists = x_str.split(symb_special[0])
    list_of_Obj = []
    for s in items_as_lists:
        item = s.strip().split()
        assert(len(item)==3)
        list_of_Obj.append(Obj(shape=item[0], color=item[1], size=item[2]))
    return list_of_Obj

def fdata_to_x_long(list_of_Obj):
    # This is the inverse of x_to_fdata.
    # Input
    #    list_of_Obj : list of Obj, each with .shape, .color, .size attributes
    # Output:
    #        x : list of strings, e.g. ['triangle', 'yellow', 'size_1', 'OBJ_SEP', 'circle', 'blue', 'size_2']
    x = []
    for i, obj in enumerate(list_of_Obj):
        x += [obj.shape, obj.color, obj.size]
        if i < len(list_of_Obj) - 1:
            x += symb_special  # adds 'OBJ_SEP'
    return x

def fdata_to_x_short(list_of_Obj):
    # This is the inverse of x_to_fdata for the short format.
    # Input
    #    list_of_Obj : list of Obj, each with .shape, .color, .size attributes
    # Output:
    #        x : list of strings, e.g. ['triangle_yellow_size_1', 'circle_blue_size_2']
    x = []
    for obj in list_of_Obj:
        x.append(f"{obj.shape}_{obj.color}_{obj.size.replace('size_', 'size')}")
    return x

def sample_set_objects(nobj_min=1, nobj_max=5):
# Sample uniformly from all possible objects without replacement
	nobj = random.randint(nobj_min,nobj_max)
	set_objs = []
	x = []
	for j in range(nobj):
		obj = [random.choice(symb_shapes),random.choice(symb_colors),random.choice(symb_sizes)]
		while obj in set_objs: # make sure we don't duplicate
			obj = [random.choice(symb_shapes),random.choice(symb_colors),random.choice(symb_sizes)]
		set_objs.append(obj)
		x += obj
		if j != nobj-1:
			x += symb_special
	return x

def generate_hypothesis():
    # sample from prior, without too many nodes
    while True:
        h = grammar.generate() 
        h = MyHypothesis(grammar=grammar,value=h)
        if h.value.count_nodes() < h.maxnodes:
            return h

def eval_hyp(x,h):
    # Evaluate the logical hypothesis on the data
    # Input
    # x : list of symbols, with different objects separated as OBJ_SEP
    # Output
    # y : list of 'yes'/'no'
    x_list_of_obj = x_to_fdata(x)
    y_bool = [h(item) for item in x_list_of_obj]
    assert( all( [isinstance(y,bool) for y in y_bool] ) )
    y = []
    for b in y_bool:
        if b:
            y.append('yes')
        else:
            y.append('no')
    return y

def forward_predictive_model(nsamp_hyp, nsamp_support, nsamp_query):
    # Draw samples from the forward model
    #
    # INPUT
    #  nsamp_hyp : number of hypotheses to be sample from the prior
    #
    # RETURN
    #  list_H : [nsamp length python list] of sampled hypotheses (each a python list of numbers)
    samples = []
    for i in range(nsamp_hyp):

        # sample from p(h)
        h = generate_hypothesis()

        # sample objects x from independent distribution 
        xs = [sample_set_objects() for _ in range(nsamp_support)]
        xq = [sample_set_objects() for _ in range(nsamp_query)]

        # sample from p(y|h;x)
        ys = [eval_hyp(myset,h) for myset in xs]
        yq = [eval_hyp(myset,h) for myset in xq]
        S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h)}}
        samples.append(S)
    return samples

def debug_fdata_x_conversion_long():
    
    # Test fdata_to_x_long is the inverse of x_to_fdata
    for _ in range(100):
        x = sample_set_objects()
        objs = x_to_fdata(x)
        x_reconstructed = fdata_to_x_long(objs)
        assert x == x_reconstructed, f"Round-trip failed:\n  {x}\n  {x_reconstructed}"
    print("fdata_to_x_long: all round-trip tests passed (100/100)")

    # Test a specific example
    x = ['triangle', 'yellow', 'size_1', 'OBJ_SEP', 'circle', 'blue', 'size_3']
    objs = x_to_fdata(x)
    assert objs[0].shape == 'triangle' and objs[0].color == 'yellow' and objs[0].size == 'size_1'
    assert objs[1].shape == 'circle' and objs[1].color == 'blue' and objs[1].size == 'size_3'
    x_back = fdata_to_x_long(objs)
    assert x == x_back
    print("fdata_to_x_long: specific example test passed")

    # Test single object (no OBJ_SEP)
    x_single = ['rectangle', 'green', 'size_2']
    objs_single = x_to_fdata(x_single)
    assert len(objs_single) == 1
    x_single_back = fdata_to_x_long(objs_single)
    assert x_single == x_single_back
    print("fdata_to_x_long: single object test passed")


def debug_fdata_x_conversion_short():
    # Test parsing of short format: ['shape_colorsize3', ...]
    # Note: short format encodes size as 'size3' (no underscore); x_to_fdata converts to 'size_3'

    # Test a specific single-object example
    x = ['circle_blue_size3']
    objs = x_to_fdata(x)
    assert len(objs) == 1
    assert objs[0].shape == 'circle' and objs[0].color == 'blue' and objs[0].size == 'size_3'
    print("debug_fdata_x_conversion_short: single object test passed")

    # Test a multi-object example
    x = ['triangle_yellow_size1', 'rectangle_green_size2', 'circle_blue_size3']
    objs = x_to_fdata(x)
    assert len(objs) == 3
    assert objs[0].shape == 'triangle' and objs[0].color == 'yellow' and objs[0].size == 'size_1'
    assert objs[1].shape == 'rectangle' and objs[1].color == 'green' and objs[1].size == 'size_2'
    assert objs[2].shape == 'circle' and objs[2].color == 'blue' and objs[2].size == 'size_3'
    print("debug_fdata_x_conversion_short: multi-object test passed")

    # Test all size variants are parsed correctly
    for i, size_str in enumerate(['size1', 'size2', 'size3'], 1):
        x = [f'circle_blue_{size_str}']
        objs = x_to_fdata(x)
        assert objs[0].size == f'size_{i}', f"Expected size_{i}, got {objs[0].size}"
    print("debug_fdata_x_conversion_short: size parsing test passed")

    # Round-trip tests: x_short -> x_to_fdata -> fdata_to_x_short -> x_short
    for _ in range(100):
        x_long = sample_set_objects()
        objs = x_to_fdata(x_long)
        x_short = fdata_to_x_short(objs)
        objs_back = x_to_fdata(x_short)
        x_short_back = fdata_to_x_short(objs_back)
        assert x_short == x_short_back, f"Round-trip failed:\n  {x_short}\n  {x_short_back}"
    print("fdata_to_x_short: all round-trip tests passed (100/100)")

    # Round-trip specific example
    x = ['triangle_yellow_size1', 'circle_blue_size3']
    objs = x_to_fdata(x)
    x_back = fdata_to_x_short(objs)
    assert x == x_back, f"Round-trip failed: {x} != {x_back}"
    print("fdata_to_x_short: specific example round-trip test passed")

    # Round-trip single object
    x_single = ['rectangle_green_size2']
    objs_single = x_to_fdata(x_single)
    x_single_back = fdata_to_x_short(objs_single)
    assert x_single == x_single_back, f"Round-trip failed: {x_single} != {x_single_back}"
    print("fdata_to_x_short: single object round-trip test passed")

if __name__ == '__main__':
    debug_fdata_x_conversion_long()
    debug_fdata_x_conversion_short()
    h = generate_hypothesis()
    xs = sample_set_objects()
    ys = eval_hyp(xs,h)
    print(h)
    print(xs)
    print(ys)
