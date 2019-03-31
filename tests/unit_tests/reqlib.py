"""
Software system requirements
"""
from collections import defaultdict

class Req(object):
	def __init__(self, req_text, parent_req=None, requirements=None):
		self.req_text = req_text
		self.parent_req = parent_req
		requirements.append(self)

	def __repr__(self):
		return "Software req: " + self.req_text

	def __hash__(self):
		return hash(self.req_text)

def swreq(req):
	"""Decorator to assign the requirement to any number of test functions 
	which will test the requirement"""
	def wrap(test_func):
		test_func.linked_req = req
		return test_func
	return wrap

def generate_traceability_matrix(requirements, test_suite, runner_output):
	scanned_classes = []
	tested_reqs = defaultdict(list)

	for ts in test_suite._tests:
		for test_class in ts._tests:
			class_name = test_class.__class__.__name__
			if class_name not in scanned_classes:
				test_methods = dir(test_class)
				for method in test_methods:
					method_fn = getattr(test_class, method)
					if hasattr(method_fn, "linked_req"):
						method_full_name = class_name + "." + method
						test_text = method_fn.__doc__
						tested_reqs[method_fn.linked_req].append((method_full_name, test_text))

			scanned_classes.append(class_name)
	
	test_output = open("test_output.html", "w")
	test_output.write(runner_output)
	test_output.write("<table>")
	test_output.write("<tr><td>Requirement</td><td>Test function</td><td>Test description</td></tr>\n")

	for req in requirements:
		if req in tested_reqs:
			for k,(method_full_name, test_text) in enumerate(tested_reqs[req]):
				if k == 0:
					output = "<tr><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (req.req_text, method_full_name, test_text)
				else:
					output = "<tr><td></td><td>%s</td><td>%s</td></tr>\n" % (method_full_name, test_text)
					
				test_output.write(output)
		else:
			test_output.write("<tr><td>%s</td><td>Missing!</td><td>Missing!</td></tr>" % req.req_text)

	test_output.write("</table>")
	test_output.close()
	