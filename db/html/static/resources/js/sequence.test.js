QUnit.test("Sequence manipulation", function(assert) {
	var sequence = new Sequence();
	assert.equal(sequence.seqlist_type(), "select");

	var generators = { 1: "gen_fn1", 2: "gen_fn2" };
	sequence.update_available_generators(generators)

	info_update1 = {3: { name: "gen_fn1:[n_targets=]", state: "saved", static: false, 
		generator:"gen_fn1", 
		params: {"param1": {"value": 4, "type": "Int", "desc": "Description"}} }};
	sequence.update(info_update1);

	assert.equal($("#seqlist").val(), 3);
	assert.equal(generators[$("#seqgen").val()], "gen_fn1");
	assert.equal(sequence.get_data(), 3); // already saved sequences just return the ID as data

	sequence.enable();

	// mock selecting the "Create New" option
	$("#seqlist").val("new")
	$("#seqlist").trigger("change");

	// check that the seqlist is now an input field which can be used to name the sequence
	assert.equal(sequence.seqlist_type(), "input");	

	// check that the name that would be made for this sequence has the generator name
	// and parameter values
	assert.equal($("#seqlist").val().startsWith("gen_fn1:[param1=4]"), true);	

	data = sequence.get_data();
	assert.equal(data['name'].startsWith("gen_fn1:[param1=4]"), true);
	assert.equal(data['generator'], 1);
	assert.equal(data['static'], false);
	assert.equal(data['params'].param1, 4);


	// update with a null dictionary, as if clicking the "Start new experiment" button
	sequence.update({})

});